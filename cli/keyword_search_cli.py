#!/usr/bin/env python3

import argparse
import json

from utils import read_stop_words, get_search_results, clean_words
from inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build inverted index")
    term_parser = subparsers.add_parser("tf", help="Get term frequency in a document")
    term_parser.add_argument("doc_id", type=int, help="Document ID")
    term_parser.add_argument("term", type=str, help="Term to search for")

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

            stopWords = read_stop_words("data/stopwords.txt")
            searchQuery = clean_words(args.query, stopWords)
            
            result_ids = set()
            for term in searchQuery:
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
