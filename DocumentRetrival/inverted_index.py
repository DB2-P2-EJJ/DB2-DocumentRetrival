import constant
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from csv import reader


class MailInvertedIndex:
    def load_data(self):
        with open(constant.DATA_FILE_NAME, 'r') as f:
            csv_reader = reader(f)
            next(csv_reader)
            for row in csv_reader:
                self._N += 1
                if sys.getsizeof(self._index) + sys.getsizeof(self._terms_frequency(row[1])):

                for (term, tf) in self._length[row[0]]:
                    if term not in self._index:
                        self._index[term] = []
                    self._index[term].append((row[0], tf))

    def __init__(self, filename):
        self._stop_words = set(stopwords.words('english'))
        self._stemmer = PorterStemmer()
        self._index = {}
        self._length = {}
        self._N = 0

    def _terms_frequency(self, text):
        tokens = [token for token in word_tokenize(text) if not token.lower() in self._stop_words]
        terms = {}
        for token in tokens:
            term = self._stemmer.stem(token)
            if term not in terms:
                terms[term] = 0
            terms[term] += 1
        return list(terms.items()).sort()

    def query(self, text, k=15):

        return []
