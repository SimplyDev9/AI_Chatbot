import os
import requests
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    HallucinationMetric
)
from deepeval.metrics.answer_relevancy import answer_relevancy
from deepeval.models import AmazonBedrockModel
from app.logger import logger


def test_rag_full_evaluation_deepeval():

    load_dotenv()

    TEST_DATA = [
        {
            "query": "Which part of the system is responsible for generating responses?",
            "expected": "LLM Layer (AWS Bedrock / Claude models)"
        },
        {
            "query": "What problem does BM25 cause in the system?",
            "expected": "It sometimes returns irrelevant keyword matches"
        }
    ]

    bedrock_model = AmazonBedrockModel(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        region=os.getenv("AWS_DEFAULT_REGION"),
        generation_kwargs={"temperature": 0}
    )

    metrics = [
        ContextualPrecisionMetric(threshold=0.5, model=bedrock_model),
        FaithfulnessMetric(threshold=0.5, model=bedrock_model),
        ContextualRelevancyMetric(threshold=0.5, model=bedrock_model),
        AnswerRelevancyMetric(threshold=0.5, model=bedrock_model),
        HallucinationMetric(threshold=0.5, model=bedrock_model),
    ]

    test_cases = []

    for item in TEST_DATA:
        query = item["query"]
        expected_output = item["expected"]


        request_payload = {
            "query": query
        }

        resp = requests.post(
            "http://127.0.0.1:8000/chat",
            json=request_payload
        )

        response = resp.json()
        logger.debug("LLM response payload: %s", response)
        print("FULL RESPONSE:", response)
        generated_output = response.get("response", "")
        retrieved_docs = response.get("retrieved_context", [])

        assert generated_output.strip() != "", "Empty LLM response"

        retrieved_texts = [
            doc.get("text", "")
            for doc in retrieved_docs
            if doc.get("text")
        ]

        assert len(retrieved_texts) > 0, "No retrieved context returned"

        test_case = LLMTestCase(
            input=query,
            actual_output=generated_output,
            expected_output=expected_output,
            retrieval_context=retrieved_texts,
            context=retrieved_texts
        )

        test_cases.append(test_case)

    results = evaluate(
        test_cases=test_cases,
        metrics=metrics
    )

    failures = []

    logger.info("--- DeepEval RAG Evaluation ---")

    for i, test_result in enumerate(results.test_results):

        query = test_result.input

        logger.info("Test Case %d | Query: %s", i + 1, query)

        for metric in test_result.metrics_data:
            status = "PASS" if metric.success else "FAIL"

            logger.info(
                "%s : %s : %s : %s",
                metric.name,
                metric.score,
                status,
                metric.reason
            )

            # Validate threshold
            if not metric.success:
                failures.append(
                    f"TestCase {i+1} | {metric.name} FAILED | Score={metric.score} | Reason={metric.reason}"
                )

    if failures:

        logger.error("===== RAG EVALUATION FAILURES =====")

        for f in failures:
            logger.error(f)

        raise AssertionError("Some RAG evaluation metrics failed")