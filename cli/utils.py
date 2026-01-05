import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def is_any_word_in_words(sourceWords, targetWords):
    for word in sourceWords:
        for tWord in targetWords:
            if word in tWord:
                return True
    return False

def read_stop_words(filePath):
    with open(filePath, 'r') as f:
        stopWords = f.read().splitlines()
    return set(stopWords)

def clean_words(words, stopWords):
    translator = str.maketrans("", "", string.punctuation)
    queryWords = words.translate(translator).lower()
    queryWords = [x for x in queryWords.split(" ") if len(x) > 0 and x not in stopWords]
    queryWords = [stemmer.stem(word) for word in queryWords]
    return queryWords

def get_search_results(query, movieContents, stopWords):
    translator = str.maketrans("", "", string.punctuation)
    result_list = []

    queryClean = query.translate(translator).lower()
    queryWords = [x for x in queryClean.split(" ") if len(x) > 0 and x not in stopWords]
    queryWords = [stemmer.stem(word) for word in queryWords]

    for movie in movieContents:
        movieTitle = movie["title"]
        movieTitleClean = movieTitle.translate(translator).lower()
        movieTitleWords = [x for x in movieTitleClean.split(" ") if len(x) > 0 and x not in stopWords]
        movieTitleWords = [stemmer.stem(word) for word in movieTitleWords]

        if is_any_word_in_words(queryWords, movieTitleWords):
            result_list.append(movieTitle)

    return result_list