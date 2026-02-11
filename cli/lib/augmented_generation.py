from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies, DEFAULT_K_PARAMETER
from lib.genai import (
    GenAIClient,
    assemble_document_query_prompt,
    assemble_summarization_prompt)


def rag_command(query: str, debug=False):
    # do RAG stuff here
    documents = load_movies()
    hybrid_search = HybridSearch(documents)

    results = hybrid_search.rrf_search(
        query,
        DEFAULT_K_PARAMETER,
        5,
        debug)

    prompt = assemble_document_query_prompt(query, results)
    genai_client = GenAIClient()
    answer = genai_client.generate_response(prompt)

    print("\nSearch Results:")
    for doc in results:
        print(f"- {doc['title']}")
    print(f"\nRAG Response:\n{answer}")


def summarize_command(query: str, limit: int):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)

    results = hybrid_search.rrf_search(
        query,
        DEFAULT_K_PARAMETER,
        limit,
        debug=False)

    prompt = assemble_summarization_prompt(query, results)
    genai_client = GenAIClient()
    summary = genai_client.generate_response(prompt)

    print("\nSearch Results:")
    for doc in results:
        print(f"- {doc['title']}")
    print(f"\nLLM Summary:\n{summary}")
