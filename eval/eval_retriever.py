# # eval_retriever.py
# from dotenv import load_dotenv

# from deepeval import evaluate
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

# from src.reranker import RerankingRetriever
# from evals.harness import load_goldens, summarize_by_metric, print_summary
# from src.retrival import build_retriever

# load_dotenv()

# GOLDEN_PATH = "golden/gold_set.json"
# JUDGE_MODEL = "gpt-4o-mini"   # NOTE: differs from the other evals (gpt-4o-mini)
# THRESHOLD = 0.7


# def run(retriever):
#     # 1. LOAD the golden set --- the fixed, human-authored truth
#     goldens = load_goldens(GOLDEN_PATH)

#     # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
#     #    then build one test case per golden.
#     test_cases = []
#     for g in goldens:
#         retrieved = retriever.invoke(g["question"])
#         retrieval_context = [doc.page_content for doc in retrieved]

#         test_cases.append(
#             LLMTestCase(
#                 input=g["question"],
#                 expected_output=g["expected_answer"],
#                 retrieval_context=retrieval_context,
#                 actual_output="(generator not evaluated in this run)",
#             )
#         )

#     # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
#     metrics = [
#         ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
#         ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
#     ]

#     # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
#     #    hyperparameters travel with the run so the report is tagged with the config.
#     result = evaluate(
#         test_cases=test_cases,
#         metrics=metrics,
#         hyperparameters={
#             "retriever": build_retriever(),          # vs "reranked" when you swap it in
#             "embedding_model": "text-embedding-3-large",
#             "chunk_size": 1000,
#             "chunk_overlap": 150,
#             "top_k": 3,
#             "judge_model": JUDGE_MODEL,
#             "golden_set": GOLDEN_PATH,
#         },
#     )
#     return summarize_by_metric(result)


# def run_local():
#     """Standalone convenience: build the retriever, then run."""
#     return run(RerankingRetriever())


# if __name__ == "__main__":
#     print_summary("retriever", run_local())

import json

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.retrival import build_retriever

load_dotenv()

GOLDEN_PATH = "golden/gold_set.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


def run(retriever):
    with open(GOLDEN_PATH, encoding="utf-8") as file:
        goldens = json.load(file)

    print(f"Loaded {len(goldens)} golden questions.", flush=True)
    test_cases = []
    for i, g in enumerate(goldens, start=1):
        print(f"Retrieving context {i}/{len(goldens)}...", flush=True)
        retrieved = retriever.invoke(g["question"])

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                expected_output=g["expected_answer"],
                retrieval_context=[doc.page_content for doc in retrieved],
                actual_output="(generator not evaluated in this run)",
            )
        )

    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True
        ),
    ]

    print("Starting judge evaluation...", flush=True)
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "vector",
            "embedding_model": "text-embedding-3-large",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 5,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
    )


def run_local():
    print("Opening or building the PDF index...", flush=True)
    retriever = build_retriever()
    print("Retriever ready.", flush=True)
    return run(retriever)


if __name__ == "__main__":
    print("Evaluation script started.", flush=True)
    run_local()
