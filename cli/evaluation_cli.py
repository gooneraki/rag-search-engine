"""
Evaluation CLI module for assessing search quality using precision, recall, and F1 score metrics.
"""
import argparse
from lib.search_utils import load_golden_dataset, load_movies
from lib.hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    golden_dataset = load_golden_dataset()

    documents = load_movies()
    hybrid_search = HybridSearch(documents)

    for test_case in golden_dataset:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

        results = hybrid_search.rrf_search(query, 60, limit)
        result_titles = [res['title'] for res in results]
        matches = [title for title in result_titles if title in relevant_docs]

        precision = len(matches)/len(results) if len(results) > 0 else 0
        recall = len(matches) / \
            len(relevant_docs) if len(relevant_docs) > 0 else 0

        f_1_score = 2 * (precision * recall) / \
            (precision + recall) if (precision + recall) > 0 else 0

        print("")
        print(f"- Query: {query}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Recall@{limit}: {recall:.4f}")
        print(f"    - F1 Score: {f_1_score:.4f}")
        print(f"    - Retrieved: {', '.join(result_titles)}")
        print(f"    - Relevant: {', '.join(relevant_docs)}")


if __name__ == "__main__":
    main()
