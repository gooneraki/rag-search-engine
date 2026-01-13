#!/usr/bin/env python3
""" Docstring for cli.semantic_search_cli """
import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text
)


def main():
    """ Docstring for main """
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    subparsers.add_parser('verify', help="Verify model")

    embed_text_parser = subparsers.add_parser('embed_text', help="Embed text")
    embed_text_parser.add_argument(
        'text', type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_query_parser = subparsers.add_parser(
        'embedquery', help="Embed query")
    embed_query_parser.add_argument(
        'query', type=str, help="Query to embed")

    args = parser.parse_args()

    match args.command:
        case 'verify':
            verify_model()

        case 'embed_text':
            embed_text(args.text)

        case 'verify_embeddings':
            verify_embeddings()

        case 'embedquery':
            embed_query_text(args.query)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
