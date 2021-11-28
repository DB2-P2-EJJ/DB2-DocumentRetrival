import os
import sys
import constant
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from csv import reader
from pathlib import Path


class MailInvertedIndex:
    # def load_data(self):
    #     with open(constant.DATA_FILE_NAME, 'r') as f:
    #         csv_reader = reader(f)
    #         next(csv_reader)
    #         for row in csv_reader:
    #             self._N += 1
    #             if sys.getsizeof(self._index) + sys.getsizeof(self._terms_frequency(row[1])):
    #
    #             for (term, tf) in self._length[row[0]]:
    #                 if term not in self._index:
    #                     self._index[term] = []
    #                 self._index[term].append((row[0], tf))

    def load_inverted_index(self, name):
        return False

    def built_inverted_index(self, name):
        data_path = os.getcwd() / Path(name + '.mii')
        index_path = data_path / Path('index')
        length_path = data_path / Path('length')
        os.makedirs(index_path)
        os.makedirs(length_path)

    def __init__(self, name):
        self._stop_words = set(stopwords.words('english'))
        self._stemmer = PorterStemmer()
        self._index = {}
        self._length = {}
        self._N = 0

        if not self.load_inverted_index(name):
            self.built_inverted_index(name)

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
        score = {}

        query_terms = self.convert_characteristic_vector(text)
        w_query = self.get_TFIDF(query_terms)

        for term in query_terms:
            list_pub = self._index[term]['pub']
            idf = self._index[term]['idf']
            for (doc_id, tf) in list_pub:
                if doc_id not in score:
                    score[doc_id] = 0
                tf_idf_doc = tf * idf
                score[doc_id] += tf_idf_doc * w_query[term]
        norm_query = self.compute_norm(query_terms)
        for doc_id in score:
            score[doc_id] = score[doc_id] / (self._norms * norm_query)
        result = sorted(score.items(), reverse=True, key=lambda tup: tup[1])
        return result[:k]
