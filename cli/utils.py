def anyWordiInWords(sourceWords, targetWords):
    for word in sourceWords:
        for tWord in targetWords:
            if word in tWord:
                return True
    return False