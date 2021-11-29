import os
import sys
import math
import constant
import pickle
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from csv import reader, field_size_limit
from pathlib import Path
from heapq import merge

field_size_limit(sys.maxsize)


class MailInvertedIndex:
    def _inverted_index_dir_exists(self):
        for directory in self._dirs:
            if not directory.is_dir():
                return False
        for file in self._files:
            if not file.is_file():
                return False
        return True

    def _load_inverted_index(self):
        if not self._inverted_index_dir_exists():
            return False
        temp = [self._N, self._n_index_block, self._n_length_block]
        for i in range(len(self._files)):
            with open(self._files[i], 'rb') as f:
                temp[i] = pickle.load(f)
        return True

    def _terms_frequency(self, text):
        tokens = [token for token in word_tokenize(text) if not token.lower() in self._stop_words]
        terms = {}
        for token in tokens:
            term = self._stemmer.stem(token)
            if term not in terms:
                terms[term] = 0
            terms[term] += 1
        result = [(k, math.log10(1 + v)) for k, v in terms.items()]
        result.sort()
        return result

    def _built_index_mail(self, mail):
        index = {}
        for (term, tf) in self._terms_frequency(mail[1]):
            if term not in index:
                index[term] = []
            index[term].append((mail[0], tf))
        return index

    def _get_index_block_path(self, i):
        return self._dirs[0] / Path("block" + str(i) + '.json')

    def _save_index_block(self):
        with open(self._get_index_block_path(self._n_index_block), 'w') as f:
            json.dump(self._index, f)
        self._index = {}
        self._n_index_block += 1

    def _add_index(self, index):
        for (t, l) in index.items():
            if t not in self._index:
                self._index[t] = l
            else:
                self._index[t] = list(merge(self._index[t], l))

    def _built_inverted_index(self):
        for directory in self._dirs:
            os.makedirs(directory)
        with open(constant.DATA_FILE_NAME, 'r') as f:
            csv_reader = reader(f)
            next(csv_reader)
            for mail in csv_reader:
                self._N += 1
                index = self._built_index_mail(mail)
                if sys.getsizeof(index) + sys.getsizeof(self._index) > constant.BLOCK_INDEX_SIZE:
                    self._save_index_block()
                self._add_index(index)

    def __init__(self):
        self._stop_words = set(stopwords.words('english') + ['subject'])
        self._stemmer = PorterStemmer()

        self._inverted_index_path = os.getcwd() / Path('email.mii')
        self._dirs = [self._inverted_index_path / p for p in [Path('index'), Path('length')]]
        self._files = [self._inverted_index_path / p for p in
                       [Path('N.bin'), Path('n_index_block.bin'), Path('n_length_block.bin')]]

        self._index = {}
        self._length = {}
        self._N = 0
        self._n_index_block = 0
        self._n_length_block = 0

        if not self._load_inverted_index():
            self._built_inverted_index()

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
