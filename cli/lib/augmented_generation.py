from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies, DEFAULT_K_PARAMETER
from lib.genai import GenAIClient


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


def assemble_document_query_prompt(query: str, docs: list[dict]) -> str:
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""

    return prompt
