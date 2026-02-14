"""Multimodal search module for generating image embeddings using CLIP models."""

from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    """Generate embeddings for images using a CLIP model.

    Embeddings are created in a vector space that is comparable with text embeddings,
    enabling semantic search across both modalities.
    """

    def __init__(self, model_name="clip-ViT-B-32"):
        """Initialize the MultimodalSearch with a specified model.

        Args:
            model_name: Name of the CLIP model to use. Defaults to clip-ViT-B-32.
        """
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str):
        """Generate an embedding for an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            A numpy array containing the image embedding.
        """
        image = Image.open(image_path)
        return self.model.encode(image)


def verify_image_embedding(image_path: str, model_name: str = "clip-ViT-B-32") -> None:
    """Verify that an image embedding can be generated and print its shape.

    Args:
        image_path: Path to the image file.
        model_name: Name of the CLIP model to use.
    """
    multi_modal_search = MultimodalSearch(model_name)
    embedding = multi_modal_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
