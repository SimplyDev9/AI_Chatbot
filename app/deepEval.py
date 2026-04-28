import os
import time
import requests
from dotenv import load_dotenv

# 🔥 CRITICAL: Control DeepEval concurrency (MUST BE BEFORE IMPORT evaluate)
os.environ["DEEPEVAL_MAX_CONCURRENCY"] = "1"
os.environ["DEEPEVAL_RETRY_ATTEMPTS"] = "5"
os.environ["DEEPEVAL_RETRY_DELAY"] = "3"

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    HallucinationMetric
)
from deepeval.models import AmazonBedrockModel
from app.logger import logger

BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/auth/login"
CHAT_URL = f"{BASE_URL}/chat"


def get_auth_token():
    login_resp = requests.post(
        LOGIN_URL,
        json={
            "email": os.getenv("TEST_EMAIL", "mahesh@test.com"),
            "password": os.getenv("TEST_PASSWORD", "Mahesh@123")
        }
    )

    try:
        login_json = login_resp.json()
    except Exception:
        raise AssertionError(f"Login response is not JSON: {login_resp.text}")

    print("LOGIN STATUS:", login_resp.status_code)
    print("LOGIN RESPONSE:", login_json)

    assert login_resp.status_code == 200, f"Login failed: {login_json}"

    token = login_json.get("access_token")
    assert token, f"access_token missing in response: {login_json}"

    return token


def test_rag_full_evaluation_deepeval():
    load_dotenv()

    TEST_DATA = [
        {
            "query": "What happens to earned leaves if an employee accumulates more than 30 ELs?",
            "expected": "If ELs are accumulated above 30 with no further utilisation, then the basic pay for "
                        "the additional leaves earned month on month will be paid on quarterly basis "
                        "in Jan, April, July, October."
        },
        {
            "query": "How many leaves are contractors and interns eligible for?",
            "expected": "Contractors and Interns are eligible for 1 CL/SL in a month."
        }
    ]

    # ✅ Stable Bedrock model (IMPORTANT FIX)
    bedrock_model = AmazonBedrockModel(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region=os.getenv("AWS_DEFAULT_REGION"),
        generation_kwargs={
            "temperature": 0,
            "maxTokens": 1500
        }
    )

    # 🔹 Metrics
    metrics = [
        ContextualPrecisionMetric(threshold=0.5, model=bedrock_model),
        FaithfulnessMetric(threshold=0.5, model=bedrock_model),
        ContextualRelevancyMetric(threshold=0.5, model=bedrock_model),
        AnswerRelevancyMetric(threshold=0.5, model=bedrock_model),
        HallucinationMetric(threshold=0.5, model=bedrock_model),
    ]

    test_cases = []

    # ✅ LOGIN
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}

    # 🔹 Build test cases
    for item in TEST_DATA:
        query = item["query"]
        expected_output = item["expected"]

        resp = requests.post(
            CHAT_URL,
            json={"query": query},
            headers=headers
        )

        response = resp.json()

        print("\nQUERY:", query)
        print("RESPONSE:", response)

        assert resp.status_code == 200, f"Chat API failed: {response}"

        generated_output = response.get("response", "")
        retrieved_docs = response.get("retrieved_context", [])

        assert generated_output.strip() != "", f"Empty LLM response: {response}"

        retrieved_texts = [
            doc.get("text", "")
            for doc in retrieved_docs
            if isinstance(doc, dict) and doc.get("text")
        ]

        if len(retrieved_texts) == 0:
            logger.warning(f"No context retrieved for query: {query}")
            continue

        test_case = LLMTestCase(
            input=query,
            actual_output=generated_output,
            expected_output=expected_output,
            retrieval_context=retrieved_texts,
            context=retrieved_texts
        )

        test_cases.append(test_case)

        time.sleep(2)  # prevent API burst

    # 🔥 RUN METRICS SAFELY
    all_failures = []

    for metric in metrics:
        print(f"\n🚀 Running Metric: {metric.__class__.__name__}")

        for test_case in test_cases:
            try:
                results = evaluate(
                    test_cases=[test_case],
                    metrics=[metric]
                )
            except Exception as e:
                logger.error(f"Evaluation failed for {test_case.input}: {str(e)}")

            for test_result in results.test_results:
                logger.info("Query: %s", test_result.input)

                for m in test_result.metrics_data:
                    status = "PASS" if m.success else "FAIL"

                    logger.info(
                        "%s : %s : %s : %s",
                        m.name,
                        m.score,
                        status,
                        m.reason
                    )

                    if not m.success:
                        all_failures.append(
                            f"{metric.__class__.__name__} | Query={test_result.input} FAILED | Score={m.score} | Reason={m.reason}"
                        )

            time.sleep(5)  # delay between test cases

        time.sleep(8)  # delay between metrics (IMPORTANT)

    if all_failures:
        logger.error("===== RAG EVALUATION FAILURES =====")
        for f in all_failures:
            logger.error(f)

        raise AssertionError("Some RAG evaluation metrics failed")
