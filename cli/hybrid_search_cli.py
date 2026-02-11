"""CLI for hybrid search combining BM25 and semantic search."""

import argparse
import json
from sentence_transformers import CrossEncoder
from lib.hybrid_search import HybridSearch, normalize_scores
from lib.search_utils import load_movies, DEFAULT_SEARCH_LIMIT, DEFAULT_K_PARAMETER
from lib.genai import (
    GenAIClient,
    prompt_spell,
    prompt_rewrite,
    prompt_expand,
    rate_movie_match,
    rate_movie_batch,
    evaluate_results)


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
        "--limit", type=int, help="Number of results to return",
        default=DEFAULT_SEARCH_LIMIT, nargs="?")

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform RRF hybrid search")
    rrf_search_parser.add_argument(
        "query", type=str, help="Search query")
    rrf_search_parser.add_argument("--limit", type=int, nargs="?",
                                   help="Number of results to return",
                                   default=DEFAULT_SEARCH_LIMIT)
    rrf_search_parser.add_argument(
        "-k", type=int, help="RRF k parameter", default=DEFAULT_K_PARAMETER, nargs="?")
    rrf_search_parser.add_argument(
        "--enhance", type=str,  choices=["spell", "rewrite", "expand"],
        help="Query enhancement method")
    rrf_search_parser.add_argument(
        "--rerank-method", nargs="?", type=str,  choices=["individual", "batch", "cross_encoder"],
        help="Reranking method to apply after initial search")
    rrf_search_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging")
    rrf_search_parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate search results with LLM")

    args = parser.parse_args()

    match args.command:
        case 'rrf-search':
            query = args.query
            genai_client = None
            debug = args.debug

            if debug:
                print("\n[DEBUG] ========== RRF SEARCH ==========")
                print(f"[DEBUG] Original Query: '{query}'")

            if args.enhance is not None or args.evaluate or \
                    (args.rerank_method is not None and args.rerank_method != "cross_encoder"):
                genai_client = GenAIClient()

            if args.enhance is not None:
                if args.enhance == "rewrite":
                    prompt = prompt_rewrite(args.query)
                elif args.enhance == "spell":
                    prompt = prompt_spell(args.query)
                elif args.enhance == "expand":
                    prompt = prompt_expand(args.query)
                else:
                    print("Enhance method not recognized.")
                    return
                query = genai_client.generate_response(prompt)

                if debug:
                    print(
                        f"[DEBUG] Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            documents = load_movies()
            hybrid_search = HybridSearch(documents)

            results = hybrid_search.rrf_search(
                query,
                args.k,
                args.limit * 5 if args.rerank_method else args.limit,
                debug=debug)

            if debug:
                print("\n[DEBUG] ========== RERANKING STAGE ==========")
            if args.rerank_method == "individual":
                for idx, res in enumerate(results):
                    score_prompt = rate_movie_match(query, res)
                    rating = genai_client.generate_response(score_prompt)
                    res['rerank_score'] = float(
                        rating.strip() if rating else 0.0)

                results.sort(key=lambda x: x['rerank_score'], reverse=True)
                results = results[:args.limit]

                if debug:
                    print("[DEBUG] Individual Rerank Results (top 5):")
                    for idx, res in enumerate(results[:5], 1):
                        print(
                            f"  {idx}. {res['title']}, Rerank Score: {res['rerank_score']:.3f}")

            if args.rerank_method == "batch":

                doc_list_str = "\n".join(
                    [f"ID: {res['id']} - Title: {res['title']} - Description: {res['description']}"
                     for res in results])
                rerank_response = genai_client.generate_response(
                    rate_movie_batch(query, doc_list_str))

                # Parse the JSON response
                ranked_ids = json.loads(rerank_response.strip())

                # Create a mapping from document ID to result
                id_to_result = {res['id']: res for res in results}

                # Assign rerank ranks based on the order in ranked_ids
                for idx, doc_id in enumerate(ranked_ids):
                    if doc_id in id_to_result:
                        id_to_result[doc_id]['rerank_rank'] = idx + 1

                # Sort by rerank_rank and limit results
                results = [id_to_result[doc_id]
                           for doc_id in ranked_ids if doc_id in id_to_result]
                results = results[:args.limit]

                if debug:
                    print("[DEBUG] Batch Rerank Results (top 5):")
                    for idx, res in enumerate(results[:5], 1):
                        print(
                            f"  {idx}. {res['title']}, Rerank Rank: {res['rerank_rank']}")

            if args.rerank_method == "cross_encoder":
                cross_encoder = CrossEncoder(
                    "cross-encoder/ms-marco-TinyBERT-L2-v2")
                pairs = []
                for doc in results:
                    pairs.append(
                        [query, f"{doc['title']} - {doc['description']}"])
                scores = cross_encoder.predict(pairs)

                for idx, doc in enumerate(results):
                    doc['cross_encoder_score'] = scores[idx]

                results.sort(
                    key=lambda x: x['cross_encoder_score'], reverse=True)

                if debug:
                    print(
                        f"[DEBUG] Cross-Encoder Scores (all {len(results)} results):")
                    for idx, res in enumerate(results, 1):
                        print(
                            f"  {idx}. {res['title']}, Score: {res['cross_encoder_score']:.3f}")

                results = results[:args.limit]

                if debug:
                    print("\n[DEBUG] Cross-Encoder Rerank Results (top 5):")
                    for idx, res in enumerate(results[:5], 1):
                        print(
                            f"  {idx}. {res['title']}, Cross-Encoder Score: {res['cross_encoder_score']:.3f}")
            # Evaluate results if flag is set
            evaluation_scores = None
            if args.evaluate:
                formatted_results: list[str] = [
                    res['title'] for res in results]
                evaluation_prompt = evaluate_results(query, formatted_results)
                evaluation_response = genai_client.generate_response(
                    evaluation_prompt)
                evaluation_scores = json.loads(evaluation_response.strip())
            for idx, res in enumerate(results, start=1):
                title_with_score = f"{res['title']}: {evaluation_scores[idx-1]}/3" \
                    if evaluation_scores else res['title']
                print(f"{idx}. {title_with_score}")
                if not evaluation_scores:
                    if args.rerank_method == "individual":
                        print(f"   Rerank Score: {res['rerank_score']:.3f}")
                    elif args.rerank_method == "batch":
                        print(f"   Rerank Rank: {res['rerank_rank']}")
                    elif args.rerank_method == "cross_encoder":
                        print(
                            f"   Cross-Encoder Score: {res['cross_encoder_score']:.3f}")
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
