import httpx
import sys
import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import os

# add the parent directory to the path to import tema_3_evaluation modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tema_3_evaluation.groq_llm import GroqDeepEval
from dotenv import load_dotenv

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

# load environment variables
load_dotenv()

BASE_URL = "http://localhost:8000"

class TestRootEndpoint:
    """Test suite for the root endpoint"""

    def test_root_endpoint_returns_json_message(self):
        """Test that root endpoint returns JSON with message key"""
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{BASE_URL}/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert isinstance(data["message"], str)
            assert len(data["message"]) > 0


class TestChatEndpointBasic:
    """Basic tests for the chat endpoint"""

    def test_chat_endpoint_accepts_valid_message(self):
        """Test that chat endpoint accepts a valid message"""
        with httpx.Client(timeout=60) as client:
            payload = {"message": "What is the due date to resolve a critical vulnerability in a system?"}
            response = client.post(f"{BASE_URL}/chat/", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "critical" in data["response"].lower()

class TestChatEndpointLLMJudge:
    """Tests for chat endpoint evaluated by LLM as a Judge"""

    @pytest.mark.asyncio
    async def test_chat_completeness_with_llm_judge(self):
        """Test that chat responses are complete and informative using LLM as a Judge"""
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BASE_URL}/chat/",
                json={"message": "What is the due date to resolve a critical vulnerability in a system?"}
            )

            assert response.status_code == 200
            data = response.json()
            actual_response = data.get("response", "")

            # Create test case for LLM evaluation
            test_case = LLMTestCase(
                input="What is the due date to resolve a critical vulnerability in a system?",
                actual_output=actual_response
            )

            # Initialize LLM judge
            groq_model = GroqDeepEval()
            evaluator = GEval(
                name="Completeness",
                criteria="Does the response provide a complete and informative answer about the due date to resolve a critical vulnerability in a system?",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                model=groq_model,
            )

            # Measure response quality
            evaluator.measure(test_case)

            # Assert that response quality meets minimum threshold
            assert evaluator.score >= 0.7, f"Response completeness score: {evaluator.score}, reason: {evaluator.reason}"

    @pytest.mark.asyncio
    async def test_chat_relevance_with_llm_judge(self):
        """Test that chat responses are relevant using LLM as a Judge"""
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BASE_URL}/chat/",
                json={"message": "Tell me about DevOps maturity models"}
            )

            assert response.status_code == 200
            data = response.json()
            actual_response = data.get("response", "")

            # Create test case for LLM evaluation
            test_case = LLMTestCase(
                input="Tell me about DevOps maturity models",
                actual_output=actual_response
            )

            # Initialize LLM judge
            groq_model = GroqDeepEval()
            evaluator = GEval(
                name="Relevance",
                criteria="Is the response relevant to the question about DevOps maturity models and does it address the user's query?",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                model=groq_model,
            )

            # Measure response quality
            evaluator.measure(test_case)

            # Assert that response relevance meets minimum threshold
            assert evaluator.score >= 0.7, f"Response relevance score: {evaluator.score}, reason: {evaluator.reason}"


class TestChatEndpointNegativeScenarios:
    """Negative test scenarios for chat endpoint evaluated by LLM as a Judge"""

    @pytest.mark.asyncio
    async def test_chat_with_out_of_scope_query_judged_by_llm(self):
        """Test that LLM judge evaluates response to out-of-scope query"""
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BASE_URL}/chat/",
                json={"message": "What is the recipe for Italian pasta?"}
            )

            assert response.status_code == 200
            data = response.json()
            actual_response = data.get("response", "")

            # Create test case for LLM evaluation
            test_case = LLMTestCase(
                input="What is the recipe for Italian pasta?",
                actual_output=actual_response
            )

            # Initialize LLM judge
            groq_model = GroqDeepEval()
            evaluator = GEval(
                name="Scope_Appropriateness",
                criteria="Is the response appropriately handling a query that is outside the scope of DevOps/cloud documentation? It should acknowledge the limitation.",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                model=groq_model,
            )

            # Measure response quality
            evaluator.measure(test_case)

            # Print the result for documentation
            print(f"Scope appropriateness score for out-of-scope query: {evaluator.score}, Reason: {evaluator.reason}")

# Test execution configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
