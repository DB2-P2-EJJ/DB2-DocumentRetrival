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
        temp = [self.N, self.n_index_block, self.n_length_block]
        for index in range(len(self._files)):
            with open(self._files[index], 'rb') as f:
                temp[index] = pickle.load(f)
        self.N, self.n_index_block, self.n_length_block = temp[0], temp[1], temp[2]
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

    def _get_block(self, directory, bp):
        block_path = self._inverted_index_path / Path(directory) / Path('block' + str(bp) + '.json')
        block = {}
        if block_path.is_file():
            with open(block_path) as f:
                block = json.load(f)
        return block

    def _save_block(self, directory, bp, block):
        block_path = self._inverted_index_path / Path(directory) / Path('block' + str(bp) + '.json')
        with open(block_path, 'w') as f:
            json.dump(block, f, indent=4, sort_keys=True)

    def _built_index_mail(self, mail):
        index = {}
        for (term, tf) in self._terms_frequency(mail[1]):
            if term not in index:
                index[term] = []
            index[term].append((int(mail[0]), tf))
        return index

    def _built_block_index(self):
        with open(constant.DATA_FILE_NAME, 'r') as f:
            csv_reader = reader(f)
            next(csv_reader)
            temp_index = {}
            for mail in csv_reader:
                self.N += 1
                index = self._built_index_mail(mail)
                for (t, ld) in index.items():
                    temp_index[t] = ld if t not in temp_index else list(merge(temp_index[t], ld))
                    if sys.getsizeof(temp_index) > constant.BLOCK_INDEX_SIZE:
                        self._save_block('index', self.n_index_block, self._index)
                        self.n_index_block += 1
                        temp_index = {t: ld}
                        self._index = {}
                    self._index[t] = ld if t not in self._index else list(merge(self._index[t], ld))
        self._save_block('index', self.n_index_block, self._index)
        self.n_index_block += 1
        self._index = {}

    def _move_block_from_temp_index_to_index(self, i, j):
        for k in range(i, j + 1):
            block = self._get_block('temp_index', k)
            self._save_block('index', k, block)

    def _merge_blocks(self, i_1, j_1, i_2, j_2):
        wp, p1, p2 = i_1, i_1, i_2
        kp1, kp2 = 0, 0
        b3, temp_b3 = {}, {}
        while p1 <= j_1 and p2 <= j_2:
            b1, b2 = self._get_block('index', p1), self._get_block('index', p2)
            keys_b1, keys_b2 = sorted(b1.keys()), sorted(b2.keys())
            while kp1 < len(keys_b1) and kp2 < len(keys_b2):
                if keys_b1[kp1] == keys_b2[kp2]:
                    key = keys_b1[kp1]
                    ld = list(merge(b1[key], b2[key]))
                    kp1 += 1
                    kp2 += 1
                else:
                    b = b1 if keys_b1[kp1] < keys_b2[kp2] else b2
                    key = min(keys_b1[kp1], keys_b2[kp2])
                    ld = b[key]
                    if keys_b1[kp1] < keys_b2[kp2]:
                        kp1 += 1
                    else:
                        kp2 += 1
                temp_b3[key] = ld
                if sys.getsizeof(temp_b3) > constant.BLOCK_INDEX_SIZE:
                    self._save_block('temp_index', wp, b3)
                    wp += 1
                    temp_b3 = {key: ld}
                    b3 = {}
                b3[key] = ld
            if kp1 >= len(keys_b1):
                p1 += 1
                kp1 = 0
            if kp2 >= len(keys_b2):
                p2 += 1
                kp2 = 0

        for p, j, kp in [(p1, j_1, kp1), (p2, j_2, kp2)]:
            while p <= j:
                b = self._get_block('index', p)
                keys = sorted(b.keys())
                for k in range(kp, len(keys)):
                    key = keys[k]
                    ld = list(merge(temp_b3[key], b[key])) if key in temp_b3 else b[key]
                    temp_b3[key] = ld
                    if sys.getsizeof(temp_b3) > constant.BLOCK_INDEX_SIZE:
                        self._save_block('temp_index', wp, b3)
                        wp += 1
                        temp_b3 = {key: ld}
                        b3 = {}
                    b3[key] = ld
                p += 1
        self._save_block('temp_index', wp, b3)
        wp += 1

        while wp <= j_2:
            self._save_block('temp_index', wp, {})
            wp += 1
        self._move_block_from_temp_index_to_index(i_1, j_2)

    def _block_sorting(self, i, j):
        if i < j:
            mid = (j + i) // 2
            self._block_sorting(i, mid)
            self._block_sorting(mid + 1, j)
            self._merge_blocks(i, mid, mid + 1, j)

    def _documents_normalization(self):
        with open(constant.DATA_FILE_NAME, 'r') as f:
            csv_reader = reader(f)
            next(csv_reader)
            temp_length = {}
            for mail in csv_reader:
                terms = self._terms_frequency(mail[1])
                for i in range(len(terms)):
                    t = list(terms[i])
                    ter = self._get_term_frequencies(terms[i][0])
                    t[1] *= 1 if len(ter) == 0 else math.log10(self.N / len(ter))
                    terms[i] = tuple(t)
                    tfidf = [t[1] for t in terms]
                    temp_length[int(mail[0])] = np.linalg.norm(tfidf)
                    if sys.getsizeof(temp_length) > constant.BLOCK_INDEX_SIZE:
                        self._save_block('length', self.n_length_block, self._length)
                        self.n_length_block += 1
                        temp_length = {int(mail[0]): np.linalg.norm(tfidf)}
                        self._length = {}
                    self._length[int(mail[0])] = np.linalg.norm(tfidf)
        self._save_block('length', self.n_length_block, self._length)
        self.n_length_block += 1
        self._length = {}

    def _save_n(self):
        data = [self.N, self.n_index_block, self.n_length_block]
        i = 0
        for file in self._files:
            with open(file, 'wb') as f:
                pickle.dump(data[i], f)
                i += 1

    def _built_inverted_index(self):
        for directory in self._dirs:
            os.makedirs(directory)
        self._built_block_index()
        self._block_sorting(0, self.n_index_block - 1)
        self._documents_normalization()
        self._save_n()

    def __init__(self):
        self._stop_words = set(stopwords.words('english') + ['subject'])
        self._stemmer = PorterStemmer()

        self._inverted_index_path = os.getcwd() / Path('email.mii')
        self._dirs = [self._inverted_index_path / p for p in [Path('index'), Path('length'), Path('temp_index')]]
        self._files = [self._inverted_index_path / p for p in
                       [Path('N.bin'), Path('n_index_block.bin'), Path('n_length_block.bin')]]

        self._index = {}
        self._length = {}
        self.N = 0
        self.n_index_block = 0
        self.n_length_block = 0

        self._norms = {}

        if not self._load_inverted_index():
            self._built_inverted_index()
        # self._documents_normalization()

    def is_sorted(self, directory, n):
        for i in range(1, self.n_index_block):
            b1, b2 = self._get_block(directory, i - 1), self._get_block(directory, i)
            if not b2:
                continue
            b1_keys, b2_keys = sorted(b1.keys()), sorted(b2.keys())
            if directory == 'length':
                b1_keys, b2_keys = [int(k) for k in b1_keys], [int(k) for k in b2_keys]
            if b1_keys[-1] > b2_keys[0]:
                return False
        return True

    def _binary_search(self, x, n, direc):
        low = 0
        high = n - 1
        res = -1
        while low <= high:
            mid = (low + high) // 2
            block = self._get_block(direc, mid)
            block_keys = sorted(block.keys())
            if not block or block_keys[0] > x:
                high = mid - 1
            elif str(block_keys[-1]) < str(x):
                low = mid + 1
            else:
                res = mid
                high = mid - 1
        return res

    def _get_term_frequencies(self, term):
        bp = self._binary_search(term, self.n_index_block, "index")
        if bp == -1:
            return []
        block = self._get_block("index", bp)
        result = []
        while term in block:
            result = list(merge(result, block[term]))
            bp += 1
            block = self._get_block("index", bp)
        return result

    def _get_tfidf(self, tf):
        q_terms = [t[0] for t in tf]
        freqs = [t[1] for t in tf]
        v_query = [math.log10(self.N / len(self._get_term_frequencies(t))) for t in q_terms if
                   self._get_term_frequencies(t)]
        print("v_query", v_query)
        print("freqs", freqs)
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

    def _get_doc_normalization(self, doc):
        bp = self._binary_search(doc, self.n_length_block, "length")
        return 0.0 if bp == -1 else list(self._get_block("length", bp).values())[doc]

    def query(self, text, k=15):
        score = {}
        tf = self._terms_frequency(text)
        query_terms = [t[0] for t in tf]
        w_query = self._get_tfidf(tf)
        for term in query_terms:
            list_pub = self._get_term_frequencies(term)
            idf = len(list_pub)
            for (doc_id, tf) in list_pub:
                if doc_id not in score:
                    score[doc_id] = 0
                tf_idf_doc = tf * idf
                score[doc_id] += tf_idf_doc * w_query
        norm_query = np.linalg.norm([t[1] for t in self._terms_frequency(text)])
        for doc_id in score:
            score[doc_id] = score[doc_id] / (self._get_doc_normalization(doc_id) * norm_query)
        result = sorted(score.items(), reverse=True, key=lambda tup: tup[1])
        return result[:k]
