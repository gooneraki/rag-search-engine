"""Semantic search module for document embedding and similarity search using sentence transformers."""
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    """Semantic search engine using sentence transformers for document embedding and similarity."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list[dict] | None = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(
            movie_strings, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit=5):
        limit = max(0, int(limit))
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first.")
        similarity_scores: list[tuple[float, dict]] = []
        query_embedding = self.generate_embedding(query)
        for i, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(
                query_embedding, doc_embedding)
            similarity_scores.append((similarity_score, self.documents[i]))

        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                'score': x[0],
                'title': x[1]['title'],
                'description': x[1]['description']
            } for x in similarity_scores[:limit]]


def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def embed_text(text):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def search_command(query, limit):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    search_results = search_instance.search(query, limit)
    for idx, res in enumerate(search_results, start=1):
        print(f"{idx}. {res['title']} (score: {res['score']:.4f})")
        desc = res['description']
        suffix = "..." if len(desc) > 120 else ""
        print(f"   {desc[:120]}{suffix}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

        if i + chunk_size >= len(words):
            break

        i += chunk_size - overlap

    print(f"Chunking {len(text)} characters")
    for idx, chunk in enumerate(chunks, start=1):
        print(f"{idx}. {chunk}")


def semantic_chunk_text(text: str, max_chunk_size, overlap):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not re.search(r"[.!?]$", text):
        sentences = [text]

    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentence = sentence.strip()
        if cleaned_sentence:
            cleaned_sentences.append(cleaned_sentence)

    if not cleaned_sentences:
        return []

    chunks = []
    i = 0
    while i < len(cleaned_sentences):
        chunk_sentences = cleaned_sentences[i:i + max_chunk_size]
        chunk = ' '.join(chunk_sentences).strip()
        if chunk:
            chunks.append(chunk)

        if i + max_chunk_size >= len(cleaned_sentences):
            break

        i += max_chunk_size - overlap

    print(f"Semantically chunking {len(text)} characters")
    for idx, chunk in enumerate(chunks, start=1):
        print(f"{idx}. {chunk}")

    return chunks
