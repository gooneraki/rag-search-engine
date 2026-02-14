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

    def rrf_search(self, query, k, limit, debug=False) -> list[dict]:

        if debug:
            print("\n[DEBUG] ========== RRF SEARCH PIPELINE ==========")
            print(f"[DEBUG] Original Query: '{query}'")
            print(f"[DEBUG] K parameter: {k}, Limit: {limit}")

        # Get BM25 and semantic results (500x the limit for better coverage)
        bm25_results = self._bm25_search(query, 500 * limit)
        sem_results = self.semantic_search.search_chunks(query, 500 * limit)

        if debug:
            print("\n[DEBUG] BM25 Search Results (top 5):")
            for idx, (doc_id, score) in enumerate(bm25_results[:5], 1):
                print(
                    f"  {idx}. Doc ID: {doc_id}, Title: {self.idx.docmap[doc_id]['title']}, Score: {score:.4f}")
            print(f"[DEBUG] BM25 Total Results: {len(bm25_results)}")

            print("\n[DEBUG] Semantic Search Results (top 5):")
            for idx, res in enumerate(sem_results[:5], 1):
                doc = self.semantic_search.document_map[res['id']]
                print(
                    f"  {idx}. Doc ID: {res['id']}, Title: {doc['title']}, Score: {res['score']:.4f}")
            print(f"[DEBUG] Semantic Total Results: {len(sem_results)}")

        combined_info: dict[int, dict] = {}

        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            if doc_id not in combined_info:
                combined_info[doc_id] = {
                    'document': self.idx.docmap[doc_id],
                    'bm25_rank': rank,
                    'semantic_rank': None
                }
            else:
                combined_info[doc_id]['bm25_rank'] = rank

        for rank, result in enumerate(sem_results, start=1):
            doc_id = result['id']
            if doc_id not in combined_info:
                combined_info[doc_id] = {
                    'document': self.semantic_search.document_map[doc_id],
                    'bm25_rank': None,
                    'semantic_rank': rank
                }
            else:
                combined_info[doc_id]['semantic_rank'] = rank

        results: list[dict] = []
        for doc_id, info in combined_info.items():
            rrf_sc = 0.0
            if info['bm25_rank'] is not None:
                rrf_sc += rrf_score(info['bm25_rank'], k)
            if info['semantic_rank'] is not None:
                rrf_sc += rrf_score(info['semantic_rank'], k)

            results.append({
                'id': doc_id,
                'title': info['document']['title'],
                'description': info['document']['description'],
                'bm25_rank': info['bm25_rank'],
                'semantic_rank': info['semantic_rank'],
                'rrf_score': rrf_sc
            })
        results.sort(key=lambda x: x['rrf_score'], reverse=True)

        if debug:
            print("\n[DEBUG] RRF Combined Results (top 10):")
            for idx, res in enumerate(results[:10], 1):
                print(f"  {idx}. {res['title']}")
                print(
                    f"     RRF Score: {res['rrf_score']:.6f}, BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}")
            print(f"[DEBUG] Total RRF Results Before Limit: {len(results)}")

        return results[:limit]


def normalize_scores(scores) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score
    if range_score == 0:
        return [1.0 for _ in scores]

    return [(score - min_score) / range_score for score in scores]


def hybrid_score_func(bm25_score: float, semantic_score: float, alpha: float) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)
