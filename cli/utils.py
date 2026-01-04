def anyWordiInWords(sourceWords, targetWords):
    for word in sourceWords:
        for tWord in targetWords:
            if word in tWord:
                return True
    return False

def readStopWords(filePath):
    with open(filePath, 'r') as f:
        stopWords = f.read().splitlines()
    return set(stopWords)