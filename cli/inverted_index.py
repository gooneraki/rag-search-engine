import os
import pickle

class InvertedIndex():
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
        for movie in movies:
            movie_id = movie["id"]
            movie_title = movie["title"]
            movie_description = movie["description"]

            self.docmap[movie_id] = movie
            self.__add_document(movie_id, f"{movie_title} {movie_description}")
    
    def save(self):
        folder_path = "cache"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)

        with open(os.path.join(folder_path, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)

            





