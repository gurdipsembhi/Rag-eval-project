# eval_retriever.py
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.reranker import CROSS_ENCODER, RerankingRetriever
from src.retrival import DB_DIR
from eval.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "golden/gold_set.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7
FETCH_K = 30
TOP_K = 3


def run(retriever):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for i, g in enumerate(goldens, start=1):
        print(f"\n[{i}/{len(goldens)}] {g['question']}", flush=True)
        retrieved = retriever.invoke(g["question"])
        print("Expected PDFs:", g.get("source_pdf_filenames", []), flush=True)
        print("Retrieved PDFs:", [doc.metadata["source"] for doc in retrieved], flush=True)
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                expected_output=g["expected_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker",
            "reranker_model": CROSS_ENCODER,
            "index_directory": DB_DIR,
            "embedding_model": "text-embedding-3-large",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "fetch_k": retriever.fetch_k,
            "top_k": retriever.top_k,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local():
    print(f"Opening or building {DB_DIR}; fetch_k={FETCH_K}, top_k={TOP_K}", flush=True)
    return run(RerankingRetriever(fetch_k=FETCH_K, top_k=TOP_K))


if __name__ == "__main__":
    print_summary("retriever", run_local())
