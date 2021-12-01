import os
import sys
import math
import constant
import pickle
import json
import numpy as np
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
        for index in range(len(self._files)):
            with open(self._files[index], 'rb') as f:
                temp[index] = pickle.load(f)
        return True

    def _terms_frequency(self, text):
        tokens = [token for token in word_tokenize(text) if not token.lower() in self._stop_words]
        terms = {}
        for token in tokens:
            term = self._stemmer.stem(token)
            if term not in terms:
                terms[term] = 0
            terms[term] += 1
        return sorted([(k, math.log10(1 + v)) for k, v in terms.items()])

    def _built_index_mail(self, mail):
        index = {}
        for (term, tf) in self._terms_frequency(mail[1]):
            if term not in index:
                index[term] = []
            index[term].append((mail[0], tf))
        return index

    def _get_index_block_path(self, index):
        return self._dirs[0] / Path("block" + str(index) + '.json')

    def _save_index_block(self):
        with open(self._get_index_block_path(self._n_index_block), 'w') as f:
            json.dump(self._index, f)
        self._index = {}
        self._n_index_block += 1

    def _built_block_index(self):
        with open(constant.DATA_FILE_NAME, 'r') as f:
            csv_reader = reader(f)
            next(csv_reader)
            temp_index = {}
            for mail in csv_reader:
                self._N += 1
                index = self._built_index_mail(mail)
                for (t, ld) in index.items():
                    temp_index[t] = ld if t not in temp_index else list(merge(temp_index[t], ld))
                    if sys.getsizeof(temp_index) > constant.BLOCK_INDEX_SIZE:
                        self._save_index_block()
                        temp_index = {t: ld}
                    self._index[t] = ld if t not in self._index else list(merge(self._index[t], ld))
            if self._index != {}:
                self._save_index_block()

    def _get_block(self, bp):
        block_path = self._get_index_block_path(bp)
        block = None
        if block_path.is_file():
            with open(block_path) as f:
                block = json.load(f)
        return block

    def _save_block(self, block, bp):
        with open(self._get_index_block_path(bp), 'w') as f:
            json.dump(block, f)

    def _merge_blocks(self, wp, pa, pb, bs):
        pae, pbe = pa + bs, pb + bs
        i, j = 0, 0
        while self._get_block(pa) is not None and pa < pae and self._get_block(pb) is not None and pb < pbe:
            block_a, block_b = self._get_block(pa), self._get_block(pb)
            block_c, block_c_temp = {}, {}
            key_a, key_b = sorted(block_a.keys()), sorted(block_b.keys())
            while i < len(key_a) and j < len(key_b):
                if key_a[i] == key_b[j]:
                    key = key_a[i]
                    ld = list(merge(block_a[key], block_b[key]))
                    i += 1
                    j += 1
                else:
                    b = block_a if key_a[i] < key_b[j] else block_b
                    key = min(key_a[i], key_b[j])
                    ld = b[key]
                    if key_a[i] < key_b[j]:
                        i += 1
                    else:
                        j += 1
                block_c_temp[key] = ld
                if sys.getsizeof(block_c_temp) > constant.BLOCK_INDEX_SIZE:
                    self._save_block(block_c, wp)
                    wp += 1
                    block_c_temp = {key: ld}
                    block_c = {}
                block_c[key] = ld
            if i >= len(key_a):
                pa += 1
                i = 0
            if j >= len(key_b):
                pb += 1
                j = 0
        for pointer, pointer_end in [(pa, pae), (pb, pbe)]:
            while self._get_block(pointer) is not None and pointer < pointer_end:
                self._save_block(self._get_block(pointer), wp)
                wp += 1
                pointer += 1
        return wp

    def _block_sorting(self):
        for bs in [2 ** i for i in range(math.ceil(math.log2(self._n_index_block)))]:
            wp = 0
            while wp < self._n_index_block:
                wp = self._merge_blocks(wp, wp, wp + bs, bs)

    def _built_inverted_index(self):
        for directory in self._dirs:
            os.makedirs(directory)
        self._built_block_index()
        self._block_sorting()

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

        self._norms = {}

        if not self._load_inverted_index():
            self._built_inverted_index()

    def _binary_search(self, left, right, term):
        mid = 0
        block_m = None
        while left < right:
            mid = (right - left)/2 + left
            block_m = self._get_block(mid)
            list_blocks = list(block_m.items())
            if mid > 0 and term not in block_m[mid] and term > list_blocks[-1][0]:
                left = mid + 1
                continue
            elif mid > 0 and term not in block_m[mid] and term < list_blocks[0][0]:
                right = mid - 1
                continue
        if mid > 0 and term in block_m[mid]:
            a = list(block_m[mid].items())[0][0]
            b = list(block_m[mid-1].items())[-1][0]
            if a == term and b == term:
                docs_a = list(block_m[mid].items())[0][1]
                docs_b = list(block_m[mid-1].items())[0][1]
                docs_a.append(docs_b)
                return {term: docs_a}
            c = list(block_m[mid].items())[-1][0]
            d = list(block_m[mid+1].items())[0][0]
            if c == term and d == term:
                docs_c = list(block_m[mid].items())[-1][1]
                docs_d = list(block_m[mid + 1].items())[0][1]
                docs_c.append(docs_d)
                return {term: docs_c}
            return {term: block_m[mid][term]}
        else:
            return None

    def _get_tfidf(self, tf):
        q_terms = [t[0] for t in tf]
        freqs = [t[1] for t in tf]
        v_query = [np.log10(self._N / len(self._binary_search(0, self._N, t)) for t in q_terms)]
        return np.dot(freqs, v_query)

    def _get_length_block_path(self, index):
        return self._dirs[1] / Path("block" + str(index) + '.json')

    def _get_length_block(self, bp):
        block_path = self._get_length_block_path(bp)
        block = None
        if block_path.is_file():
            with open(block_path) as f:
                block = json.load(f)
        return block

    def _get_norms(self, index):
        # pendiente
        return

    def _compute_norm(self, vector):
        v = np.array(vector)
        return np.linalg.norm(v)

    def query(self, text, k=15):
        score = {}
        tf = self._terms_frequency(text)
        query_terms = [t[0] for t in tf]
        w_query = self._get_tfidf(tf)
        for term in query_terms:
            list_pub = self._index[term]['pub']
            idf = self._index[term]['idf']
            for (doc_id, tf) in list_pub:
                if doc_id not in score:
                    score[doc_id] = 0
                tf_idf_doc = tf * idf
                score[doc_id] += tf_idf_doc * w_query[term]
        norm_query = self._compute_norm(query_terms)
        index_norms = self._norms
        # self._get_norms()
        for doc_id in score:
            score[doc_id] = score[doc_id] / (index_norms * norm_query)
        result = sorted(score.items(), reverse=True, key=lambda tup: tup[1])
        return result[:k]