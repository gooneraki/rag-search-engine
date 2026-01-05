#!/usr/bin/env python3

import argparse
import json

from utils import read_stop_words, get_search_results
from inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            searchQuery = args.query

            stopWords = read_stop_words("data/stopwords.txt")
            with open("data/movies.json") as f:
                data = json.load(f)
                movieContents = data["movies"]
            
            result_list = get_search_results(searchQuery,movieContents,stopWords)

            for i,result in enumerate(result_list[:5]):
                print(f"{i+1}. {result}")

        case "build":
            print("Building inverted index...")
            with open("data/movies.json") as f:
                data = json.load(f)
                movieContents = data["movies"]
            index = InvertedIndex()
            index.build(movieContents)
            index.save()
            docs = index.get_documents('merida')
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
