from sentence_transformers import SentenceTransformer


class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text: str):
        """ Docstring for generate_embedding """
        if len(text.strip()) == 0:
            raise ValueError("You need to provide a text")
        embeddings = self.model.encode([text])

        return embeddings[0]


def verify_model():
    semanticSearch = SemanticSearch()

    print(f"Model loaded: {semanticSearch.model}")
    print(f"Max sequence length: {semanticSearch.model.max_seq_length}")


def embed_text(text):
    """ Docstring for embed_text """
    semanticSearch = SemanticSearch()
    embedding = semanticSearch.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
