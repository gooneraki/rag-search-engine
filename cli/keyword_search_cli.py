#!/usr/bin/env python3
"""
Docstring for cli.keyword_search_cli
"""
import argparse
import math

from utils import read_stop_words, clean_words
from inverted_index import InvertedIndex, tokenize_text


def main() -> None:
    """
    Docstring for main
    """
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # build_parser = subparsers.add_parser("build", help="Build inverted index")
    term_parser = subparsers.add_parser(
        "tf", help="Get term frequency in a document")
    term_parser.add_argument("doc_id", type=int, help="Document ID")
    term_parser.add_argument("term", type=str, help="Term to search for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency of a term")
    idf_parser.add_argument("term", type=str, help="Term to search for")

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

            results = sorted(list(result_ids))[:5]
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

            tokenized_term = tokenize_text(args.term)
            if len(tokenized_term) > 1:
                print("Error: Please provide a single term for IDF calculation.")
                return
            elif len(tokenized_term) == 0:
                print(
                    f"Error: No valid term provided after tokenization for '{args.term}'.")
                return

            term = tokenized_term[0]
            doc_ids = index.get_documents(term)
            total_doc_count = len(index.docmap)
            term_match_doc_count = len(doc_ids)

            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
