""" Docstring for cli.lib.semantic_search """
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import CACHE_DIR, load_movies


class SemanticSearch():
    """ Docstring for SemanticSearch """

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        """ Docstring for generate_embedding """
        if len(text.strip()) == 0:
            raise ValueError("You need to provide a text")
        embeddings = self.model.encode([text])

        return embeddings[0]

    def build_embeddings(self, documents: list[dict]):
        """ Docstring for build_embeddings """
        self.documents = documents
        all_doc_strings = []
        for doc_el in documents:
            self.document_map[doc_el['id']] = doc_el
            doc_string = f"{doc_el['title']}: {doc_el['description']}"
            all_doc_strings.append(doc_string)

        self.embeddings = self.model.encode(
            all_doc_strings, show_progress_bar=True)

        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "movie_embeddings.npy"), "wb") as f:
            np.save(f, self.embeddings)

    def load_or_create_embeddings(self, documents: list[dict]):
        """ Docstring for load_or_create_embeddings """
        self.documents = documents

        for doc_el in documents:
            self.document_map[doc_el['id']] = doc_el

        if (os.path.exists(os.path.join(CACHE_DIR, "movie_embeddings.npy"))):
            with open(os.path.join(CACHE_DIR, "movie_embeddings.npy"), "rb") as f:
                self.embeddings = np.load(f)
            if (len(self.embeddings) == len(documents)):
                return self.embeddings

        else:
            self.embeddings = self.build_embeddings(documents)


def verify_model():
    """ Docstring for verify_model """
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    """ Docstring for embed_text """
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    """ Docstring for verify_embeddings """
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(embeddings)}")

    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
