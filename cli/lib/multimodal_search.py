"""Multimodal search module for generating image embeddings using CLIP models."""

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .semantic_search import cosine_similarity
from .search_utils import load_movies


class MultimodalSearch:
    """Generate embeddings for images using a CLIP model.

    Embeddings are created in a vector space that is comparable with text embeddings,
    enabling semantic search across both modalities.
    """

    def __init__(self, documents=None, model_name="clip-ViT-B-32"):
        """Initialize the MultimodalSearch with a specified model.

        Args:
            documents: Optional list of movie documents to create text embeddings for.
            model_name: Name of the CLIP model to use. Defaults to clip-ViT-B-32.
        """
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = None
        self.text_embeddings = None

        if documents is not None:
            # Create texts by concatenating title and description
            self.texts = [
                f"{doc['title']}: {doc['description']}"
                for doc in documents
            ]
            # Generate text embeddings
            self.text_embeddings = self.model.encode(
                self.texts, show_progress_bar=True
            )

    def embed_image(self, image_path: str):
        """Generate an embedding for an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            A numpy array containing the image embedding.
        """
        image = Image.open(image_path)
        return self.model.encode(image)

    def search_with_image(self, image_path: str):
        """Search for similar movies using an image.

        Args:
            image_path: Path to the image file to search with.

        Returns:
            A list of dicts containing the top 5 most similar documents,
            each with id, title, description, and similarity score.
        """
        if self.text_embeddings is None:
            raise ValueError(
                "No text embeddings available. Initialize with documents.")

        # Generate embedding for the image
        image_embedding = self.embed_image(image_path)

        # Calculate cosine similarity between image and all text embeddings
        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append({
                'id': self.documents[i]['id'],
                'title': self.documents[i]['title'],
                'description': self.documents[i]['description'],
                'similarity': similarity
            })

        # Sort by similarity score in descending order and return top 5
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:5]


def image_search_command(image_path: str):
    """Search for movies similar to the provided image.

    Args:
        image_path: Path to the image file to search with.

    Returns:
        A list of dicts containing the top 5 most similar movies.
    """
    documents = load_movies()
    multimodal_search = MultimodalSearch(documents)
    results = multimodal_search.search_with_image(image_path)
    return results


def verify_image_embedding(image_path: str, model_name: str = "clip-ViT-B-32") -> None:
    """Verify that an image embedding can be generated and print its shape.

    Args:
        image_path: Path to the image file.
        model_name: Name of the CLIP model to use.
    """
    multi_modal_search = MultimodalSearch(model_name)
    embedding = multi_modal_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
