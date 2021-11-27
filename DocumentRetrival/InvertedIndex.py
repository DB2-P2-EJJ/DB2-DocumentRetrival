from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class InvertedIndex:

    def load_index(self, filename):
        return {}

    def __init__(self, name):
        self._index = self.load_index(name)
        self._stop_words = set(stopwords.words('english'))
        self._stemmer = PorterStemmer()

    def convert_characteristic_vector(self, text):
        tokens = [token for token in word_tokenize(text) if not token.lower() in self._stop_words]
        terms = {}
        for token in tokens:
            term = self._stemmer.stem(token)
            if term not in term:
                terms[term] = 0
            terms[term] += 1
        return list(terms.items()).sort()

    def add_email(self, email):

        return None

    def query(self, text, k=15):
        return []
