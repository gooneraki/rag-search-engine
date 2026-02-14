"""CLI for Retrieval Augmented Generation (RAG) operations."""
import argparse

from lib.augmented_generation import (
    rag_command,
    summarize_command,
    citations_command,
    question_command)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results")
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, nargs="?",
        help="Number of search results to use for summarization (default: 5)")

    citations_parser = subparsers.add_parser(
        "citations", help="Generate citations for search results")
    citations_parser.add_argument(
        "query", type=str, help="Search query for generating citations")
    citations_parser.add_argument(
        "--limit", type=int, default=5, nargs="?",
        help="Number of search results to use for generating citations (default: 5)")

    question_parser = subparsers.add_parser(
        "question", help="Answer a question using search results")
    question_parser.add_argument(
        "question", type=str, help="Question to answer based on search results")
    question_parser.add_argument(
        "--limit", type=int, default=5, nargs="?",
        help="Number of search results to use for answering (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "question":
            question_command(args.question, args.limit)

        case "rag":
            query = args.query
            rag_command(query)

        case "summarize":
            query = args.query
            limit = args.limit
            summarize_command(query, limit)

        case "citations":
            citations_command(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
