class RemoveWord:

    def __init__(self, list_of_words=None):
        self.list_of_words = list_of_words

    # Save list of stopwords to file
    def save(self, filepath):
        with open(filepath, "w") as f:
            for word in list_of_words:
                f.write("{0}\n".format(word))

    # Load a stopwords file
    def load(self, filepath):
        self.list_of_words = open(filepath).read().splitlines()
