from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self, name):
        self._index = {}
        self._name = name

    def addDocument(self, document):

        return None

    def query(self, text, k=15):
        return []
