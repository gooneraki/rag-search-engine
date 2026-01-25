"""Inverted index implementation for keyword search with BM25 ranking."""
import os
import pickle
import string
import math
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from lib.search_utils import (
    BM25_K1,
    BM25_B,
    CACHE_DIR,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    """Inverted index for efficient keyword search with TF-IDF and BM25 scoring."""

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.doc_lengths: dict[int, int] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.term_frequencies_path = os.path.join(
            CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        """Add document to index with tokenization and term frequency tracking."""
        tokens = tokenize_text(text)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        """Calculate average document length for BM25 normalization."""
        return sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0.0

    def build(self) -> None:
        """Build index from loaded movie documents."""
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        """Persist index and metadata to cache directory."""
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        """Load index and metadata from cache."""
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        """Get document IDs containing a term."""
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        """Get term frequency in a document."""
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term."""
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        """Calculate TF-IDF score for a term in a document."""
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        """Calculate BM25 inverse document frequency."""
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])

        return math.log((doc_count - term_doc_count + 0.5) /
                        (term_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        """Calculate BM25 term frequency with length normalization."""

        tf = self.get_tf(doc_id, term)

        avg_doc = self.__get_avg_doc_length()
        length_norm = 1 - b + b * \
            ((self.doc_lengths[doc_id] / avg_doc) if avg_doc > 0 else 0)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term):
        """Calculate BM25 score for a term in a document."""
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return idf * tf

    def bm25_search(self, query, limit):
        """Search documents using BM25 ranking."""
        tokens = tokenize_text(query)
        scores = defaultdict(float)

        for doc_id in self.docmap.keys():
            for token in tokens:
                scores[doc_id] += self.bm25(doc_id, token)

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:limit]


def preprocess_text(text: str) -> str:
    """Lowercase and remove punctuation from text."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    """Preprocess and tokenize text, filtering stopwords."""
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


def bm25_idf_command(term: str) -> float:
    """Calculate and return BM25 IDF for a term."""
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    return index.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B):
    """Calculate and return BM25 TF for a term in a document."""
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        bm25_tf = index.get_bm25_tf(doc_id, term, k1, b)
        return bm25_tf
    except KeyError:
        print(f"Document ID {doc_id} not found.")
        return 0.0


def bm25_search_command(query, limit):
    """Execute BM25 search and return formatted results."""
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        search_results = index.bm25_search(query, limit)
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            doc_id = result[0]
            score = result[1]
            title = index.docmap[doc_id]['title']
            formatted_results.append(
                f"{i}. ({doc_id}) {title} - Score: {score:.2f}")
        return formatted_results
    except Exception as err:
        raise ValueError(f"Error in bm25search: {err}") from err
