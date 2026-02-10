"""Utility functions and constants for search operations."""
import json
import os

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_K_PARAMETER = 60

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
GOLDEN_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75


def load_movies() -> list[dict]:
    """Load movie data from JSON file."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    """Load stopwords from text file."""
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]
