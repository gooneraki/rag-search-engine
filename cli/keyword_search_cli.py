#!/usr/bin/env python3

import argparse
import json
import string


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    translator = str.maketrans("", "", string.punctuation)

    match args.command:
        case "search":
            searchQuery = args.query
            print(f"Searching for: {searchQuery}")
            
            result_list = []

            with open("data/movies.json") as f:
                data = json.load(f)
                movieContents = data["movies"]

            for movie in movieContents:
                movieTitle = movie["title"]
                if searchQuery.translate(translator).lower() in movieTitle.translate(translator).lower():
                    result_list.append(movieTitle)

            for i,result in enumerate(result_list[:5]):
                print(f"{i+1}. {result}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
