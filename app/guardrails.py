"""
app/guardrails.py
─────────────────
Enterprise-grade, multi-layer content guardrail pipeline.

Execution order (all layers run sequentially; first block wins):
─────────────────────────────────────────────────────────────────
  Layer 1 — Comprehend detect_toxic_content()
              PROFANITY · HATE_SPEECH · INSULT · SEXUAL · VIOLENCE_OR_THREAT · GRAPHIC

  Layer 2 — Comprehend detect_pii_entities()
              SSN · CREDIT_CARD · BANK_ACCOUNT · PASSPORT · DRIVING_LICENSE ·
              EMAIL · PHONE · ADDRESS · DATE_OF_BIRTH · AWS_ACCESS_KEY · PASSWORD ·
              and 20+ more entity types

  Layer 3 — AWS Bedrock apply_guardrail()
              Jailbreak / prompt-attack detection · Denied topics ·
              Custom word filters · Contextual grounding (output side)

Design principles
─────────────────
  • Fail-open   — an AWS outage NEVER blocks a legitimate user request.
                  All three layers catch boto3/botocore exceptions and return clean.
  • Structured  — every block returns (True, user_facing_message).
                  Callers raise HTTP 400 with { code, message }; FE reads it uniformly.
  • Auditable   — every block is WARNING-logged with label, score, and a 120-char preview.
  • Tunable     — thresholds and PII entity lists are env-var / constant overrides;
                  no code change needed to add a new denied PII type.

Usage (in any FastAPI route)
────────────────────────────
    from app.guardrails import run_guardrails

    is_blocked, user_message = run_guardrails(user_text)
    if is_blocked:
        raise HTTPException(
            status_code=400,
            detail={"code": "CONTENT_BLOCKED", "message": user_message},
        )
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# AWS clients
# ─────────────────────────────────────────────────────────────────────────────
_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY")

_comprehend = boto3.client(
    "comprehend",
    region_name=_REGION,
    aws_access_key_id=_KEY_ID,
    aws_secret_access_key=_SECRET,
)

_bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=_REGION,
    aws_access_key_id=_KEY_ID,
    aws_secret_access_key=_SECRET,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (override via .env)
# ─────────────────────────────────────────────────────────────────────────────

# Layer 1 — Toxicity threshold [0.0–1.0].  0.7 recommended for enterprise.
COMPREHEND_TOXICITY_THRESHOLD: float = float(
    os.environ.get("COMPREHEND_TOXICITY_THRESHOLD", "0.7")
)

# Layer 2 — PII confidence threshold [0.0–1.0].  0.8 avoids false positives.
COMPREHEND_PII_THRESHOLD: float = float(
    os.environ.get("COMPREHEND_PII_THRESHOLD", "0.8")
)

# Layer 2 — Which PII entity types to BLOCK outright.
# Full list: https://docs.aws.amazon.com/comprehend/latest/dg/how-pii.html
# WARN_ONLY types are logged but not blocked (useful for audit-only mode).
PII_BLOCK_TYPES: frozenset[str] = frozenset(
    os.environ.get(
        "PII_BLOCK_TYPES",
        (
            "SSN,"
            "CREDIT_DEBIT_NUMBER,"
            "CREDIT_DEBIT_CVV,"
            "CREDIT_DEBIT_EXPIRY,"
            "BANK_ACCOUNT_NUMBER,"
            "BANK_ROUTING,"
            "PASSPORT_NUMBER,"
            "DRIVER_ID,"
            "PIN,"
            "PASSWORD,"
            "AWS_ACCESS_KEY,"
            "AWS_SECRET_KEY,"
            "IP_ADDRESS,"
            "MAC_ADDRESS"
        ),
    ).split(",")
)

# These PII types are logged for audit but NOT blocked (less sensitive).
PII_WARN_ONLY_TYPES: frozenset[str] = frozenset(
    os.environ.get(
        "PII_WARN_ONLY_TYPES",
        "EMAIL,PHONE,ADDRESS,DATE_OF_BIRTH,AGE,URL,NAME",
    ).split(",")
)

# Layer 3 — Bedrock Guardrail
BEDROCK_GUARDRAIL_ID: str = os.environ.get("BEDROCK_GUARDRAIL_ID", "")
BEDROCK_GUARDRAIL_VERSION: str = os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT")


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable label maps
# ─────────────────────────────────────────────────────────────────────────────

_TOXICITY_LABEL_FRIENDLY: dict[str, str] = {
    "PROFANITY":          "profanity",
    "HATE_SPEECH":        "hate speech",
    "SEXUAL":             "sexual content",
    "VIOLENCE_OR_THREAT": "violent or threatening language",
    "INSULT":             "insulting language",
    "GRAPHIC":            "graphic content",
}

_PII_LABEL_FRIENDLY: dict[str, str] = {
    "SSN":                  "a Social Security Number",
    "CREDIT_DEBIT_NUMBER":  "a credit or debit card number",
    "CREDIT_DEBIT_CVV":     "a card CVV code",
    "CREDIT_DEBIT_EXPIRY":  "a card expiry date",
    "BANK_ACCOUNT_NUMBER":  "a bank account number",
    "BANK_ROUTING":         "a bank routing number",
    "PASSPORT_NUMBER":      "a passport number",
    "DRIVER_ID":            "a driver's license number",
    "PIN":                  "a PIN",
    "PASSWORD":             "a password",
    "AWS_ACCESS_KEY":       "an AWS access key",
    "AWS_SECRET_KEY":       "an AWS secret key",
    "IP_ADDRESS":           "an IP address",
    "MAC_ADDRESS":          "a MAC address",
    "EMAIL":                "an email address",
    "PHONE":                "a phone number",
    "ADDRESS":              "a physical address",
    "DATE_OF_BIRTH":        "a date of birth",
    "NAME":                 "a personal name",
    "AGE":                  "age information",
    "URL":                  "a URL",
}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — AWS Comprehend: Toxicity
# ─────────────────────────────────────────────────────────────────────────────

def _check_toxicity(text: str) -> Tuple[bool, str]:
    """
    Call Comprehend detect_toxic_content.

    Returns (True, label_name) if any label exceeds COMPREHEND_TOXICITY_THRESHOLD.
    Returns (False, "")        if clean or service unavailable (fail-open).
    """
    try:
        response = _comprehend.detect_toxic_content(
            TextSegments=[{"Text": text[:5_000]}],
            LanguageCode="en",
        )

        for result in response.get("ResultList", []):
            for label in result.get("Labels", []):
                score: float = label.get("Score", 0.0)
                name: str = label.get("Name", "UNKNOWN")

                if score >= COMPREHEND_TOXICITY_THRESHOLD:
                    logger.warning(
                        "GUARDRAIL BLOCK | layer=toxicity | label=%s score=%.3f | preview=%r",
                        name, score, text[:120],
                    )
                    return True, name

        return False, ""

    except (BotoCoreError, ClientError) as exc:
        logger.error("Comprehend toxicity unavailable — failing open: %s", exc)
        return False, ""
    except Exception as exc:
        logger.exception("Unexpected toxicity error — failing open: %s", exc)
        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — AWS Comprehend: PII Detection
# ─────────────────────────────────────────────────────────────────────────────

def _check_pii(text: str) -> Tuple[bool, str, list[dict]]:
    """
    Call Comprehend detect_pii_entities on the user's input.

    Behaviour
    ─────────
    • Entity types in PII_BLOCK_TYPES  and score >= PII_THRESHOLD → BLOCK
    • Entity types in PII_WARN_ONLY_TYPES                         → LOG only
    • Everything else                                             → ignore

    Returns
    -------
    (True,  entity_type, detected_list)   — blocked; entity_type is the first hit
    (False, "",          detected_list)   — clean or warn-only; list for audit logging
    """
    try:
        response = _comprehend.detect_pii_entities(
            Text=text[:100_000],   # Comprehend hard limit for detect_pii_entities
            LanguageCode="en",
        )

        entities: list[dict] = response.get("Entities", [])
        detected_for_audit: list[dict] = []

        for entity in entities:
            etype: str  = entity.get("Type", "")
            score: float = entity.get("Score", 0.0)
            begin: int  = entity.get("BeginOffset", 0)
            end: int    = entity.get("EndOffset", 0)

            # Mask the actual value in logs (never log raw PII)
            masked_value = f"[{etype}:{begin}-{end}]"

            if score < COMPREHEND_PII_THRESHOLD:
                continue  # Below confidence threshold — ignore

            if etype in PII_BLOCK_TYPES:
                logger.warning(
                    "GUARDRAIL BLOCK | layer=pii | type=%s score=%.3f | position=%d-%d",
                    etype, score, begin, end,
                )
                detected_for_audit.append({"type": etype, "score": score, "masked": masked_value})
                return True, etype, detected_for_audit

            if etype in PII_WARN_ONLY_TYPES:
                logger.info(
                    "GUARDRAIL AUDIT | layer=pii | type=%s score=%.3f (warn-only, not blocked)",
                    etype, score,
                )
                detected_for_audit.append({"type": etype, "score": score, "masked": masked_value, "warn_only": True})

        return False, "", detected_for_audit

    except (BotoCoreError, ClientError) as exc:
        logger.error("Comprehend PII unavailable — failing open: %s", exc)
        return False, "", []
    except Exception as exc:
        logger.exception("Unexpected PII error — failing open: %s", exc)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper — parse Bedrock assessment to distinguish BLOCK vs ANONYMIZE
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bedrock_intervention(response: dict) -> dict:
    """
    Inspect a Bedrock apply_guardrail response and return a structured summary
    of what the guardrail actually did.

    Bedrock conflates all intervention types under action="GUARDRAIL_INTERVENED".
    We need to distinguish:

      HARD BLOCK  — topic policy, content filter, word filter, prompt attack,
                    contextual grounding failure, or PII with action=BLOCKED.
                    The response MUST NOT reach the user.

      SOFT ANONYMIZE — PII or regex entities with action=ANONYMIZED.
                    Bedrock has already replaced the sensitive values in
                    outputs[0]["text"] with tokens like [NAME], [EMAIL].
                    This text IS safe to use.

    Returns
    -------
    {
        "has_hard_block": bool,
        "has_anonymization": bool,
        "anonymized_text": str,         # outputs[0]["text"] when anonymized
        "triggered_policies": list[str] # human-readable list for logging
    }
    """
    result = {
        "has_hard_block":    False,
        "has_anonymization": False,
        "anonymized_text":   "",
        "triggered_policies": [],
    }

    outputs = response.get("outputs", [])
    if outputs:
        result["anonymized_text"] = outputs[0].get("text", "")

    for assessment in response.get("assessments", []):

        # ── Topic policy (denied topics) — always a hard block ─────────────
        for topic in assessment.get("topicPolicy", {}).get("topics", []):
            if topic.get("action") == "BLOCKED":
                result["has_hard_block"] = True
                result["triggered_policies"].append(
                    f"DENIED_TOPIC:{topic.get('name', 'unknown')}"
                )

        # ── Content policy (hate, violence, prompt attack) — always BLOCKED ─
        for f in assessment.get("contentPolicy", {}).get("filters", []):
            if f.get("action") == "BLOCKED":
                result["has_hard_block"] = True
                result["triggered_policies"].append(
                    f"CONTENT_FILTER:{f.get('type', 'unknown')}"
                )

        # ── Word policy (managed + custom word lists) — always BLOCKED ──────
        word_policy = assessment.get("wordPolicy", {})
        for w in word_policy.get("customWords", []):
            if w.get("action") == "BLOCKED":
                result["has_hard_block"] = True
                result["triggered_policies"].append("WORD_FILTER:CUSTOM")
        for w in word_policy.get("managedWordLists", []):
            if w.get("action") == "BLOCKED":
                result["has_hard_block"] = True
                result["triggered_policies"].append("WORD_FILTER:MANAGED")

        # ── Sensitive information policy — can be ANONYMIZED *or* BLOCKED ───
        sensitive = assessment.get("sensitiveInformationPolicy", {})
        for entity in sensitive.get("piiEntities", []):
            entity_action = entity.get("action", "")
            entity_type   = entity.get("type", "UNKNOWN")
            if entity_action == "ANONYMIZED":
                result["has_anonymization"] = True
                result["triggered_policies"].append(f"PII_ANONYMIZED:{entity_type}")
            elif entity_action == "BLOCKED":
                result["has_hard_block"] = True
                result["triggered_policies"].append(f"PII_BLOCKED:{entity_type}")
        for regex in sensitive.get("regexes", []):
            regex_action = regex.get("action", "")
            regex_name   = regex.get("name", "CUSTOM_REGEX")
            if regex_action == "ANONYMIZED":
                result["has_anonymization"] = True
                result["triggered_policies"].append(f"REGEX_ANONYMIZED:{regex_name}")
            elif regex_action == "BLOCKED":
                result["has_hard_block"] = True
                result["triggered_policies"].append(f"REGEX_BLOCKED:{regex_name}")

        # ── Contextual grounding (hallucination / relevance) — always BLOCKED
        for f in assessment.get("contextualGroundingPolicy", {}).get("filters", []):
            if f.get("action") == "BLOCKED":
                result["has_hard_block"] = True
                result["triggered_policies"].append(
                    f"GROUNDING:{f.get('type', 'unknown')} "
                    f"score={f.get('score', '?'):.3f} "
                    f"threshold={f.get('threshold', '?')}"
                )

    # Safety net: GUARDRAIL_INTERVENED with no parsed policies
    # means Bedrock added a new policy type we do not know about yet.
    # Treat as hard block to be safe.
    if (not result["has_hard_block"] and not result["has_anonymization"]
            and response.get("action") == "GUARDRAIL_INTERVENED"):
        result["has_hard_block"] = True
        result["triggered_policies"].append("UNKNOWN_POLICY")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — AWS Bedrock Guardrail — INPUT side
# ─────────────────────────────────────────────────────────────────────────────

def _check_bedrock_guardrail(text: str) -> Tuple[bool, str]:
    """
    Apply the Bedrock Guardrail to the raw USER INPUT.

    Handles: jailbreaks, prompt attacks, denied topics, custom word filters,
    and sensitive information policies.

    ANONYMIZE behaviour on input
    ────────────────────────────
    If Bedrock anonymizes input PII (e.g. replaces a name in the query with
    [NAME]) the request is ALLOWED through. Comprehend (Layer 2) already
    blocked the genuinely dangerous PII types; anything reaching here and
    being anonymized by Bedrock is low-risk (e.g. a name in a sentence).
    We log it and continue — the query proceeds with the original text because
    the LLM needs full context to answer correctly. The OUTPUT guardrail
    (check_output_guardrail) will anonymize any PII in the response.

    Returns
    -------
    (True,  user_message)  — hard block detected; raise HTTP 400
    (False, "")            — clean, or only soft anonymization; proceed
    """
    if not BEDROCK_GUARDRAIL_ID:
        return False, ""

    try:
        response = _bedrock_runtime.apply_guardrail(
            guardrailIdentifier=BEDROCK_GUARDRAIL_ID,
            guardrailVersion=BEDROCK_GUARDRAIL_VERSION,
            source="INPUT",
            content=[{"text": {"text": text}}],
        )

        if response.get("action") != "GUARDRAIL_INTERVENED":
            return False, ""

        parsed = _parse_bedrock_intervention(response)

        if parsed["has_hard_block"]:
            logger.warning(
                "GUARDRAIL BLOCK | layer=bedrock_input | id=%s | policies=%s | preview=%r",
                BEDROCK_GUARDRAIL_ID,
                ",".join(parsed["triggered_policies"]),
                text[:120],
            )
            return True, (
                "Your message was flagged by our content policy and cannot be processed. "
                "Please rephrase your question and try again."
            )

        # Soft anonymization only — log for audit, allow through
        if parsed["has_anonymization"]:
            logger.info(
                "GUARDRAIL AUDIT | layer=bedrock_input | action=ANONYMIZED | "
                "policies=%s | query continues unchanged",
                ",".join(parsed["triggered_policies"]),
            )

        return False, ""

    except (BotoCoreError, ClientError) as exc:
        logger.error("Bedrock input guardrail unavailable — failing open: %s", exc)
        return False, ""
    except Exception as exc:
        logger.exception("Unexpected Bedrock input guardrail error — failing open: %s", exc)
        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Public API — run_guardrails()
# ─────────────────────────────────────────────────────────────────────────────

def run_guardrails(text: str) -> Tuple[bool, str]:
    """
    Run the full 3-layer enterprise guardrail pipeline against raw user input.

    Pipeline
    ────────
    Layer 1  Comprehend detect_toxic_content   (profanity, hate, violence, etc.)
    Layer 2  Comprehend detect_pii_entities    (SSN, card numbers, passwords, etc.)
    Layer 3  Bedrock apply_guardrail INPUT     (jailbreak, denied topics, custom policy)

    First layer to block wins; subsequent layers are not called.
    All layers fail open on AWS service errors.

    Parameters
    ----------
    text : str
        Raw user input, already stripped and length-validated by the route.

    Returns
    -------
    (False, "")             — clean; proceed to answer_query()
    (True,  user_message)   — blocked; raise HTTP 400 { code, message }
    """
    if not text or not text.strip():
        return False, ""

    # ── Layer 1: Toxicity ────────────────────────────────────────────────
    blocked, label = _check_toxicity(text)
    if blocked:
        friendly = _TOXICITY_LABEL_FRIENDLY.get(label, "inappropriate content")
        return True, (
            f"Your message contains {friendly} and cannot be processed. "
            "Please keep the conversation respectful and professional."
        )

    # ── Layer 2: PII ─────────────────────────────────────────────────────
    blocked, pii_type, detected = _check_pii(text)

    warn_only = [e for e in detected if e.get("warn_only")]
    if warn_only:
        logger.info(
            "GUARDRAIL AUDIT | warn-only PII in message | types=%s",
            [e["type"] for e in warn_only],
        )

    if blocked:
        friendly = _PII_LABEL_FRIENDLY.get(pii_type, "sensitive personal information")
        return True, (
            f"Your message appears to contain {friendly}. "
            "Please do not share sensitive personal or financial information in this chat. "
            "Remove the sensitive data and try again."
        )

    # ── Layer 3: Bedrock Guardrail ────────────────────────────────────────
    blocked, _ = _check_bedrock_guardrail(text)
    if blocked:
        return True, (
            "Your message was flagged by our content policy and cannot be processed. "
            "Please rephrase your question and try again."
        )

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Output-side guardrail  (called from chatbot.py AFTER the LLM responds)
# ─────────────────────────────────────────────────────────────────────────────

def check_output_guardrail(llm_response: str) -> Tuple[bool, str]:
    """
    Apply the Bedrock Guardrail to the LLM's OUTPUT before sending to the user.

    This catches:
      • Hallucinated responses contradicting your knowledge base (grounding check)
      • PII reproduced from retrieved documents (e.g. names, emails in KB docs)
      • Any policy violation in the generated text

    ANONYMIZE vs BLOCK — the key distinction fixed here
    ────────────────────────────────────────────────────
    When Bedrock anonymizes PII in the output (action=ANONYMIZED), it has
    already produced a safe version of the response with tokens like
    [NAME], [EMAIL], [PHONE] replacing the actual values.
    We USE that anonymized text instead of blocking the response entirely.

    Example:
        LLM produces:  "Sarah Johnson approved your leave request."
        Bedrock output: "[NAME] approved your leave request."
        We return:      "[NAME] approved your leave request."   ← user sees this

    Only HARD BLOCKS (topic policy, content filter, grounding failure,
    PII with action=BLOCKED) suppress the response and return a fallback.

    Returns
    -------
    (False, "")                     — response is safe; send original to user
    (False, anonymized_text)        — response was anonymized; send anonymized to user
    (True,  fallback_message)       — hard block; send fallback_message instead
    """
    if not BEDROCK_GUARDRAIL_ID:
        return False, ""

    if not llm_response or not llm_response.strip():
        return False, ""

    try:
        response = _bedrock_runtime.apply_guardrail(
            guardrailIdentifier=BEDROCK_GUARDRAIL_ID,
            guardrailVersion=BEDROCK_GUARDRAIL_VERSION,
            source="OUTPUT",
            content=[{"text": {"text": llm_response}}],
        )

        if response.get("action") != "GUARDRAIL_INTERVENED":
            return False, ""

        parsed = _parse_bedrock_intervention(response)

        # ── HARD BLOCK: suppress response, return fallback ────────────────
        if parsed["has_hard_block"]:
            logger.warning(
                "GUARDRAIL BLOCK | layer=bedrock_output | id=%s | policies=%s | preview=%r",
                BEDROCK_GUARDRAIL_ID,
                ",".join(parsed["triggered_policies"]),
                llm_response[:120],
            )
            return True, (
                "I was unable to generate a safe response for that query. "
                "Please try rephrasing your question."
            )

        # ── SOFT ANONYMIZE: use the anonymized text Bedrock already produced ─
        if parsed["has_anonymization"]:
            anonymized = parsed["anonymized_text"]
            if anonymized:
                logger.info(
                    "GUARDRAIL ANONYMIZE | layer=bedrock_output | id=%s | "
                    "policies=%s | PII replaced in response",
                    BEDROCK_GUARDRAIL_ID,
                    ",".join(parsed["triggered_policies"]),
                )
                # Return (False, anonymized_text) — not a block, but use the
                # cleaned text. The caller (chatbot.py) must check the second
                # value: if non-empty, use it; if empty, use the original.
                return False, anonymized

            # Bedrock said ANONYMIZED but gave us no output text — fail safe,
            # return the original (PII may not actually be present).
            logger.warning(
                "GUARDRAIL ANOMALY | layer=bedrock_output | "
                "ANONYMIZED flag set but no output text returned — using original",
            )
            return False, ""

        return False, ""

    except (BotoCoreError, ClientError) as exc:
        logger.error("Bedrock output guardrail unavailable — failing open: %s", exc)
        return False, ""
    except Exception as exc:
        logger.exception("Unexpected output guardrail error — failing open: %s", exc)
        return False, ""