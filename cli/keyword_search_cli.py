#!/usr/bin/env python3
"""CLI for keyword-based search using BM25 algorithm."""
import argparse

from lib.search_utils import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT
from lib.utils import read_stop_words, clean_words
from lib.keyword_search import InvertedIndex, bm25_idf_command, bm25_tf_command, bm25_search_command


def main() -> None:
    """Entry point for keyword search """
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build inverted index")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    term_parser = subparsers.add_parser(
        "tf", help="Get term frequency in a document")
    term_parser.add_argument("doc_id", type=int, help="Document ID")
    term_parser.add_argument("term", type=str, help="Term to search for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency of a term")
    idf_parser.add_argument("term", type=str, help="Term to search for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument(
        "b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit", type=float, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Max number of results")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            stop_words = read_stop_words("data/stopwords.txt")
            search_query = clean_words(args.query, stop_words)

            result_ids = set()
            for term in search_query:
                doc_ids = index.get_documents(term)
                result_ids.update(doc_ids)

            results = sorted(list(result_ids))[:DEFAULT_SEARCH_LIMIT]
            for doc_id in results:
                movie = index.docmap[doc_id]
                print(f"{movie['title']} (ID: {doc_id})")

        case "build":
            print("Building inverted index...")
            index = InvertedIndex()
            index.build()
            index.save()

        case "idf":
            index = InvertedIndex()

            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            idf = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tf":
            index = InvertedIndex()

            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            try:
                tf = index.get_tf(args.doc_id, args.term)
                print(tf)
            except KeyError:
                print("0")

        case "tfidf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            tf_idf = index.get_tf_idf(args.doc_id, args.term)

            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":

            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(
                args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            results = bm25_search_command(args.query, args.limit)
            for result in results:
                print(result)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
