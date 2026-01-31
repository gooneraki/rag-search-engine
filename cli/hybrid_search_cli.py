"""CLI for hybrid search combining BM25 and semantic search."""

import argparse
from lib.hybrid_search import HybridSearch, normalize_scores
from lib.search_utils import load_movies
from lib.genai import GenAIClient, prompt_for_typo


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize scores from different search methods")
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search")
    weighted_search_parser.add_argument(
        "query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha", type=float, help="Weight for BM25 search", default=0.5, nargs="?")
    weighted_search_parser.add_argument(
        "--limit", type=int, help="Number of results to return", default=5, nargs="?")

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform RRF hybrid search")
    rrf_search_parser.add_argument(
        "query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "--limit", type=int, help="Number of results to return", default=5, nargs="?")
    rrf_search_parser.add_argument(
        "-k", type=int, help="RRF k parameter", default=60, nargs="?")
    rrf_search_parser.add_argument(
        "--enhance", type=str,  choices=["spell"], help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case 'rrf-search':
            query = args.query
            if args.enhance == "spell":
                genai_client = GenAIClient()
                typo_prompt = prompt_for_typo(args.query)
                query = genai_client.generate_response(typo_prompt)

                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query, args.k, args.limit)

            for idx, res in enumerate(results, start=1):
                print(f"{idx}. {res['title']}")
                print(f"   RRF Score: {res['rrf_score']:.3f}")
                print(
                    f"   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}")
                desc = res['description']
                suffix = "..." if len(desc) > 100 else ""
                print(f"   {desc[:100]}{suffix}")

        case 'weighted-search':
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.weighted_search(
                args.query, args.alpha, args.limit)

            for idx, res in enumerate(results, start=1):
                print(f"{idx}. {res['title']}")
                print(f"   Hybrid Score: {res['hybrid_score']:.3f}")
                print(
                    f"   BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                desc = res['description']
                suffix = "..." if len(desc) > 100 else ""
                print(f"   {desc[:100]}{suffix}")

        case "normalize":
            scores = args.scores
            if not scores:
                return

            normalized_scores = normalize_scores(scores)

            for score in normalized_scores:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
