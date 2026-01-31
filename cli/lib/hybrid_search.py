"""Hybrid search module combining BM25 and semantic search."""

import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    """Hybrid search combining BM25 keyword search with semantic search."""

    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5) -> list[dict]:

        # Get BM25 and semantic results (500x the limit for better coverage)
        bm25_results = self._bm25_search(query, 500 * limit)
        sem_results = self.semantic_search.search_chunks(query, 500 * limit)

        # Normalize scores from both search methods
        bm25_scores = [score for _, score in bm25_results]
        bm25_norm = normalize_scores(bm25_scores) if bm25_scores else []

        sem_scores = [result['score'] for result in sem_results]
        sem_norm = normalize_scores(sem_scores) if sem_scores else []

        # Create a dictionary mapping document IDs to combined information
        combined_info: dict[int, dict] = {}

        # Add BM25 results with normalized scores
        for i, (doc_id, _) in enumerate(bm25_results):
            if doc_id not in combined_info:
                combined_info[doc_id] = {
                    'document': self.idx.docmap[doc_id],
                    'bm25_score': bm25_norm[i],
                    'semantic_score': 0.0
                }
            else:
                combined_info[doc_id]['bm25_score'] = bm25_norm[i]

        # Add semantic search results with normalized scores
        for i, result in enumerate(sem_results):
            doc_id = result['id']
            if doc_id not in combined_info:
                combined_info[doc_id] = {
                    'document': self.semantic_search.document_map[doc_id],
                    'bm25_score': 0.0,
                    'semantic_score': sem_norm[i]
                }
            else:
                combined_info[doc_id]['semantic_score'] = sem_norm[i]

        # Calculate hybrid score for each document and sort
        results = []
        for doc_id, info in combined_info.items():
            hybrid_score = hybrid_score_func(
                info['bm25_score'],
                info['semantic_score'],
                alpha
            )
            results.append({
                'id': doc_id,
                'title': info['document']['title'],
                'description': info['document']['description'],
                'bm25_score': info['bm25_score'],
                'semantic_score': info['semantic_score'],
                'hybrid_score': hybrid_score
            })

        # Sort by hybrid score in descending order and return top results
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score
    if range_score == 0:
        return [1.0 for _ in scores]

    return [(score - min_score) / range_score for score in scores]


def hybrid_score_func(bm25_score: float, semantic_score: float, alpha: float) -> float:
    """
    Calculate hybrid score as a weighted combination of BM25 and semantic scores.

    Args:
        bm25_score: Normalized BM25 score (0-1)
        semantic_score: Normalized semantic score (0-1)
        alpha: Weight for BM25 (0-1). Semantic weight is (1-alpha)

    Returns:
        Hybrid score combining both signals
    """
    return alpha * bm25_score + (1 - alpha) * semantic_score
