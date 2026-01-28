"""CLI for hybrid search combining BM25 and semantic search."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize scores from different search methods")
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="Scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if not scores:
                return

            float_scores = scores

            min_score = min(float_scores)
            max_score = max(float_scores)
            range_score = max_score - min_score
            if range_score == 0:
                normalized_scores = [1.0 for _ in float_scores]
            else:
                normalized_scores = [
                    (score - min_score) / range_score for score in float_scores
                ]

            for score in normalized_scores:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
