from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from groq_llm import GroqDeepEval
from report import save_report
import sys
from dotenv import load_dotenv
import httpx
import asyncio

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
THRESHOLD = 0.8

test_cases = [
    # ToDo: Adăugați un scenariu care să fie evaluat de LLM as a Judge
    LLMTestCase(
        input="What is the due date to resolve a critical vulnerability in a system?"
    ),
    # ToDo: Adăugați un scenariu care să fie evaluat de LLM as a Judge
    LLMTestCase(
        input="What are the typical timelines for remediation of security vulnerabilities?."
    ),
]

groq_model = GroqDeepEval()

evaluator1 = GEval(
    # ToDo: Adăugați numele metricii și criteriul de evaluare.
    name="Relevancy",
    criteria="""
    Evaluate if the answer is relevant to any of the following criteria:
    
    - The answer should directly address the question about the due date for resolving a critical vulnerability.
    - The answer should mention specific timeframes (e.g., 30 days, 60 days) or guidelines provided by industry standards (e.g., CVSS, NIST).
    - The answer should reference best practices for vulnerability management and remediation timelines.
    - The answer should indicate the importance of timely resolution for critical vulnerabilities to prevent exploitation.
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

evaluator2 = GEval(
    # ToDo: Adăugați numele metricii și criteriul de evaluare.
    name="Correctness",
    criteria="""
    Evaluate if the answer is correct or incorrect based on the following criteria:
    
    - The answer should provide accurate information about the typical due dates for resolving critical vulnerabilities, which are often recommended to be within 30 to 60 days, depending on the severity and context.
    - The answer should reference established guidelines or standards for vulnerability management, such as those from CVSS or NIST, which may suggest specific timelines for remediation based on the criticality of the vulnerability.
    - The answer should emphasize the importance of prioritizing critical vulnerabilities for timely resolution to mitigate potential risks and prevent exploitation by attackers.
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)


async def _fetch_response(client: httpx.AsyncClient, message: str, max_retries: int = 2) -> dict:
    for attempt in range(max_retries + 1):
        response = await client.post(f"{BASE_URL}/chat/", json={"message": message})
        data = response.json()
        if data.get("detail") != "The chat response has expired":
            return data
        if attempt < max_retries:
            await asyncio.sleep(2)
    return data


async def _run_evaluation() -> tuple[list[dict], list[float], list[float]]:
    results: list[dict] = []
    scores1: list[float] = []
    scores2: list[float] = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        for i, case in enumerate(test_cases, 1):
            candidate = await _fetch_response(client, case.input)
            case.actual_output = candidate

            evaluator1.measure(case)
            evaluator2.measure(case)

            print(f"[{i}/{len(test_cases)}] {case.input[:60]}...")
            # ToDo: Personalizați afișarea scorurilor pentru fiecare metrică.
            print(f"Relevancy: {evaluator1.score:.2f} | Correctness: {evaluator2.score:.2f}")

            results.append({
                "input": case.input,
                "response": candidate.get("response", str(candidate)) if isinstance(candidate, dict) else str(candidate),
                # ToDo: Adăugați în dicționar scorurile și motivele pentru fiecare metrică.
                "Relevancy_score": evaluator1.score,
                "Relevancy_reason": evaluator1.reason,
                "Correctness_score": evaluator2.score,
                "Correctness_reason": evaluator2.reason,
            })
            scores1.append(evaluator1.score)
            scores2.append(evaluator2.score)

    return results, scores1, scores2


def run_evaluation() -> None:
    results, scores1, scores2 = asyncio.run(_run_evaluation())
    output_file = save_report(results, scores1, scores2, THRESHOLD)
    print(f"\nReport saved in: {output_file}")


if __name__ == "__main__":
    run_evaluation()
