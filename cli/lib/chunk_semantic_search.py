import os
import json
import numpy as np
from .semantic_search import (
    SemanticSearch,
    semantic_chunk_text,
    cosine_similarity
)
from .search_utils import (CACHE_DIR, load_movies)


CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")
SCORE_PRECISION = 4


class ChunkedSemanticSearch(SemanticSearch):
    """
    Docstring for ChunkedSemanticSearch
    """

    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        """
        Build chunk embeddings from documents
        """
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        chunk_strings = []
        chunk_metadata = []
        for doc in documents:
            if (len(doc['description'].strip()) == 0):
                continue
            chunks = semantic_chunk_text(doc['description'], 4, 1)
            for i, chunk in enumerate(chunks):
                chunk_strings.append(chunk)
                chunk_metadata.append({
                    'movie_idx': doc['id'],
                    'chunk_idx': i,
                    "total_chunks": len(chunks)
                })

        self.chunk_embeddings = self.model.encode(
            chunk_strings, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, 'w') as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(
                chunk_metadata)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, 'r') as f:
                data = json.load(f)
                if len(self.chunk_embeddings) == data["total_chunks"]:
                    self.chunk_metadata = data["chunks"]
                    return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        """
        Search chunks for a given query
        """
        limit = max(0, int(limit))
        query_embedding = self.model.encode([query])[0]

        movies: dict = {}

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            movie_idx = self.chunk_metadata[i]['movie_idx']

            if movie_idx not in movies or score > movies[movie_idx]['score']:
                movies[movie_idx] = {
                    "movie": self.document_map[movie_idx],
                    "score": score
                }

        sorted_movies = sorted(
            movies.items(), key=lambda x: x[1]['score'], reverse=True)
        top_movies = sorted_movies[:limit]

        results = []
        for movie_idx, movie_data in top_movies:
            doc = movie_data['movie']
            score = movie_data['score']
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, SCORE_PRECISION),
                "metadata": doc.get("metadata", {})
            })

        return results


def embed_text_chunks():
    """
    Embed text chunks from documents
    """
    chunked_search = ChunkedSemanticSearch()

    documents = load_movies()
    embeddings = chunked_search.load_or_create_chunk_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int):
    """
    Search chunked embeddings for a given query
    """
    chunked_search = ChunkedSemanticSearch()

    documents = load_movies()
    chunked_search.load_or_create_chunk_embeddings(documents)

    results = chunked_search.search_chunks(query, limit)

    for i, result in enumerate(results, start=1):

        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
