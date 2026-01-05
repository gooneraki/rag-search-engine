import os
import pickle
from utils import clean_words, read_stop_words

class InvertedIndex():
    folder_path = "cache"
    index_file_path = os.path.join(folder_path, "index.pkl")
    docmap_file_path = os.path.join(folder_path, "docmap.pkl")
    

    index: dict[str, set[int]] = {}
    docmap: dict[int, str] = {}

    def __add_document(self, doc_id, text):
        words = text.lower().split()
        for word in words:
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)

    def get_documents(self, term):
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
        
    def build(self, movies):
        stop_words = read_stop_words("data/stopwords.txt")
        for movie in movies:
            movie_id = movie["id"]
            movie_title = movie["title"]
            movie_description = movie["description"]

            self.docmap[movie_id] = movie
            text = f"{movie_title} {movie_description}"
            cleaned_words = clean_words(text, stop_words)
            for word in cleaned_words:
                if word not in self.index:
                    self.index[word] = set()
                self.index[word].add(movie_id)
    
    def save(self):
        
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        with open(self.index_file_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_file_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        if not os.path.exists(self.index_file_path):
            raise FileNotFoundError(f"Index file not found at {self.index_file_path}")
        if not os.path.exists(self.docmap_file_path):
            raise FileNotFoundError(f"Docmap file not found at {self.docmap_file_path}")
        
        with open(self.index_file_path, "rb") as f:
            self.index = pickle.load(f)
        
        with open(self.docmap_file_path, "rb") as f:
            self.docmap = pickle.load(f)





