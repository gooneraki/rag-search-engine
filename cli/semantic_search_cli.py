#!/usr/bin/env python3
""" Docstring for cli.semantic_search_cli """
import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_text
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

    search_query_parser = subparsers.add_parser(
        'search', help="search query")
    search_query_parser.add_argument(
        'query', type=str, help="Query to search")
    search_query_parser.add_argument(
        "--limit", type=float, nargs='?', default=5, help="Limit results")

    chunk_text_parser = subparsers.add_parser(
        'chunk', help="chunk text")
    chunk_text_parser.add_argument(
        'text', type=str, help="text to chunk")
    chunk_text_parser.add_argument(
        "--overlap", type=float, nargs='?', default=0,  help="Chunk overlap")
    chunk_text_parser.add_argument(
        "--chunk-size", type=float, nargs='?', default=200, help="Chunk size")

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

        case 'search':
            search_command(args.query, args.limit)

        case 'chunk':
            chunk_text(args.text, int(args.chunk_size), int(args.overlap))

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
