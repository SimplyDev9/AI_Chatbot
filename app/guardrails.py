"""
app/guardrails.py

Enterprise-grade, multi-layer content guardrail pipeline.

Execution order (all layers run sequentially; first block wins):
─────────────────────────────────────────────────────────────────
Layer 1 — Comprehend detect_toxic_content()
    PROFANITY · HATE_SPEECH · INSULT · SEXUAL · VIOLENCE_OR_THREAT · GRAPHIC

Layer 2 — Comprehend detect_pii_entities()
    SSN · CREDIT_CARD · BANK_ACCOUNT · PASSPORT · DRIVING_LICENSE ·
    EMAIL · PHONE · ADDRESS · DATE_OF_BIRTH · AWS_ACCESS_KEY · PASSWORD · ...

Layer 3 — AWS Bedrock apply_guardrail()
    Jailbreak / prompt-attack detection
    Denied topics
    Custom word filters
    Contextual grounding (output side)
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# =============================================================================
# AWS Clients
# =============================================================================

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

# =============================================================================
# Configuration
# =============================================================================

COMPREHEND_TOXICITY_THRESHOLD: float = float(
    os.environ.get("COMPREHEND_TOXICITY_THRESHOLD", "0.7")
)

COMPREHEND_PII_THRESHOLD: float = float(
    os.environ.get("COMPREHEND_PII_THRESHOLD", "0.8")
)

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

PII_WARN_ONLY_TYPES: frozenset[str] = frozenset(
    os.environ.get(
        "PII_WARN_ONLY_TYPES",
        "EMAIL,PHONE,ADDRESS,DATE_OF_BIRTH,AGE,URL,NAME",
    ).split(",")
)

BEDROCK_GUARDRAIL_ID: str = os.environ.get("BEDROCK_GUARDRAIL_ID", "")
BEDROCK_GUARDRAIL_VERSION: str = os.environ.get(
    "BEDROCK_GUARDRAIL_VERSION",
    "DRAFT",
)

# =============================================================================
# Friendly Labels
# =============================================================================

_TOXICITY_LABEL_FRIENDLY = {
    "PROFANITY": "profanity",
    "HATE_SPEECH": "hate speech",
    "SEXUAL": "sexual content",
    "VIOLENCE_OR_THREAT": "violent or threatening language",
    "INSULT": "insulting language",
    "GRAPHIC": "graphic content",
}

_PII_LABEL_FRIENDLY = {
    "SSN": "a Social Security Number",
    "CREDIT_DEBIT_NUMBER": "a credit or debit card number",
    "CREDIT_DEBIT_CVV": "a card CVV code",
    "CREDIT_DEBIT_EXPIRY": "a card expiry date",
    "BANK_ACCOUNT_NUMBER": "a bank account number",
    "BANK_ROUTING": "a bank routing number",
    "PASSPORT_NUMBER": "a passport number",
    "DRIVER_ID": "a driver's license number",
    "PIN": "a PIN",
    "PASSWORD": "a password",
    "AWS_ACCESS_KEY": "an AWS access key",
    "AWS_SECRET_KEY": "an AWS secret key",
    "IP_ADDRESS": "an IP address",
    "MAC_ADDRESS": "a MAC address",
    "EMAIL": "an email address",
    "PHONE": "a phone number",
    "ADDRESS": "a physical address",
    "DATE_OF_BIRTH": "a date of birth",
    "NAME": "a personal name",
    "AGE": "age information",
    "URL": "a URL",
}


# =============================================================================
# Layer 1 — Toxicity Detection
# =============================================================================

def _check_toxicity(text: str) -> Tuple[bool, str]:
    """
    Call Comprehend detect_toxic_content().

    Returns:
        (True, label_name) if blocked
        (False, "") if clean or service unavailable
    """
    try:
        response = _comprehend.detect_toxic_content(
            TextSegments=[{"Text": text[:5000]}],
            LanguageCode="en",
        )

        for result in response.get("ResultList", []):
            for label in result.get("Labels", []):
                score = label.get("Score", 0.0)
                name = label.get("Name", "UNKNOWN")

                if score >= COMPREHEND_TOXICITY_THRESHOLD:
                    logger.warning(
                        "GUARDRAIL BLOCK | layer=toxicity | "
                        "label=%s score=%.3f | preview=%r",
                        name,
                        score,
                        text[:120],
                    )
                    return True, name

        return False, ""

    except (BotoCoreError, ClientError) as exc:
        logger.error(
            "Comprehend toxicity unavailable — failing open: %s",
            exc,
        )
        return False, ""

    except Exception as exc:
        logger.exception(
            "Unexpected toxicity error — failing open: %s",
            exc,
        )
        return False, ""


# =============================================================================
# Layer 2 — PII Detection
# =============================================================================

def _check_pii(text: str) -> Tuple[bool, str, list[dict]]:
    """
    Detect PII using AWS Comprehend.
    """
    try:
        response = _comprehend.detect_pii_entities(
            Text=text[:100000],
            LanguageCode="en",
        )

        entities = response.get("Entities", [])
        detected_for_audit = []

        for entity in entities:
            etype = entity.get("Type", "")
            score = entity.get("Score", 0.0)
            begin = entity.get("BeginOffset", 0)
            end = entity.get("EndOffset", 0)

            masked_value = f"[{etype}:{begin}-{end}]"

            if score < COMPREHEND_PII_THRESHOLD:
                continue

            if etype in PII_BLOCK_TYPES:
                logger.warning(
                    "GUARDRAIL BLOCK | layer=pii | "
                    "type=%s score=%.3f | position=%d-%d",
                    etype,
                    score,
                    begin,
                    end,
                )

                detected_for_audit.append(
                    {
                        "type": etype,
                        "score": score,
                        "masked": masked_value,
                    }
                )

                return True, etype, detected_for_audit

            if etype in PII_WARN_ONLY_TYPES:
                logger.info(
                    "GUARDRAIL AUDIT | layer=pii | "
                    "type=%s score=%.3f (warn-only)",
                    etype,
                    score,
                )

                detected_for_audit.append(
                    {
                        "type": etype,
                        "score": score,
                        "masked": masked_value,
                        "warn_only": True,
                    }
                )

        return False, "", detected_for_audit

    except (BotoCoreError, ClientError) as exc:
        logger.error(
            "Comprehend PII unavailable — failing open: %s",
            exc,
        )
        return False, "", []

    except Exception as exc:
        logger.exception(
            "Unexpected PII error — failing open: %s",
            exc,
        )
        return False, "", []


# =============================================================================
# Layer 3 — Bedrock Guardrail
# =============================================================================

def _check_bedrock_guardrail(text: str) -> Tuple[bool, str]:
    """
    Apply AWS Bedrock Guardrail to user input.
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

        action = response.get("action", "NONE")

        if action == "GUARDRAIL_INTERVENED":
            triggered_policies = []

            for assessment in response.get("assessments", []):
                if assessment.get("topicPolicy", {}).get("topics"):
                    triggered_policies.append("DENIED_TOPIC")

                if assessment.get("contentPolicy", {}).get("filters"):
                    triggered_policies.append("CONTENT_FILTER")

                if assessment.get("wordPolicy"):
                    triggered_policies.append("WORD_FILTER")

                if assessment.get("sensitiveInformationPolicy"):
                    triggered_policies.append("SENSITIVE_INFO")

                if assessment.get("contextualGroundingPolicy"):
                    triggered_policies.append("GROUNDING")

            logger.warning(
                "GUARDRAIL BLOCK | layer=bedrock | id=%s | "
                "policies=%s | preview=%r",
                BEDROCK_GUARDRAIL_ID,
                ",".join(triggered_policies) or "UNKNOWN",
                text[:120],
                )

            outputs = response.get("outputs", [])
            reason = (
                outputs[0].get("text", "Content policy violation")
                if outputs
                else "Content policy violation"
            )

            return True, reason

        return False, ""

    except (BotoCoreError, ClientError):
        return False, ""

    except Exception:
        return False, ""

# =============================================================================
# Public API — run_guardrails()
# =============================================================================

def run_guardrails(text: str) -> Tuple[bool, str]:
    """
    Run the full 3-layer enterprise guardrail pipeline.

    Pipeline:
        Layer 1 - Comprehend Toxicity
        Layer 2 - Comprehend PII
        Layer 3 - Bedrock Guardrail

    Returns:
        (False, "")            -> clean
        (True, user_message)   -> blocked
    """
    if not text or not text.strip():
        return False, ""

    # -------------------------------------------------------------------------
    # Layer 1 — Toxicity
    # -------------------------------------------------------------------------
    blocked, label = _check_toxicity(text)

    if blocked:
        friendly = _TOXICITY_LABEL_FRIENDLY.get(
            label,
            "inappropriate content",
        )

        return True, (
            f"Your message contains {friendly} and cannot be processed. "
            "Please keep the conversation respectful and professional."
        )

    # -------------------------------------------------------------------------
    # Layer 2 — PII
    # -------------------------------------------------------------------------
    blocked, pii_type, detected = _check_pii(text)

    warn_only = [e for e in detected if e.get("warn_only")]

    if warn_only:
        logger.info(
            "GUARDRAIL AUDIT | warn-only PII in message | types=%s",
            [e["type"] for e in warn_only],
        )

    if blocked:
        friendly = _PII_LABEL_FRIENDLY.get(
            pii_type,
            "sensitive personal information",
        )

        return True, (
            f"Your message appears to contain {friendly}. "
            "Please do not share sensitive personal or financial "
            "information in this chat. Remove the sensitive data "
            "and try again."
        )

    # -------------------------------------------------------------------------
    # Layer 3 — Bedrock Guardrail
    # -------------------------------------------------------------------------
    blocked, _ = _check_bedrock_guardrail(text)

    if blocked:
        return True, (
            "Your message was flagged by our content policy and "
            "cannot be processed. Please rephrase your question "
            "and try again."
        )

    return False, ""


# =============================================================================
# Output-side Guardrail
# =============================================================================

def check_output_guardrail(llm_response: str) -> Tuple[bool, str]:
    """
    Apply Bedrock Guardrail to the LLM OUTPUT.
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
            content=[
                {
                    "text": {
                        "text": llm_response
                    }
                }
            ],
        )

        action = response.get("action", "NONE")

        if action == "GUARDRAIL_INTERVENED":

            assessments = response.get("assessments", [])

            only_anonymized = True

            for assessment in assessments:

                sensitive_policy = assessment.get(
                    "sensitiveInformationPolicy",
                    {}
                )

                for entity in sensitive_policy.get("piiEntities", []):

                    entity_action = entity.get("action")

                    logger.info(
                        "Output PII detected | type=%s | action=%s",
                        entity.get("type"),
                        entity_action,
                    )

                    if entity_action != "ANONYMIZED":
                        only_anonymized = False

            logger.warning(
                "GUARDRAIL INTERVENED | layer=bedrock_output | "
                "id=%s | anonymized_only=%s | preview=%r",
                BEDROCK_GUARDRAIL_ID,
                only_anonymized,
                llm_response[:120],
            )

            if only_anonymized:
                logger.info(
                    "Only anonymizable PII detected. "
                    "Allowing response to proceed."
                )
                return False, ""

            return True, (
                "I was unable to generate a safe response for that query. "
                "Please try rephrasing your question."
            )

        return False, ""

    except (BotoCoreError, ClientError) as exc:
        logger.error(
            "Bedrock output guardrail unavailable — failing open: %s",
            exc,
        )
        return False, ""

    except Exception as exc:
        logger.exception(
            "Unexpected output guardrail error — failing open: %s",
            exc,
        )
        return False, ""