
"""CLI for multimodal search operations using CLIP models."""

import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command


def main() -> None:
    """Main entry point for the multimodal search CLI."""
    parser = argparse.ArgumentParser(
        description="Multimodal Search CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding generation")
    verify_image_embedding_parser.add_argument(
        "path", type=str, help="Path to an image file")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search for movies using an image")
    image_search_parser.add_argument(
        "image_path", type=str, help="Path to an image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)

        case "image_search":
            results = image_search_command(args.image_path)
            for idx, result in enumerate(results, start=1):
                print(
                    f"{idx}. {result['title']} (similarity: {result['similarity']:.3f})")
                desc = result['description']
                suffix = "..." if len(desc) > 100 else ""
                print(f"   {desc[:100]}{suffix}")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
