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
        self._norms = {}

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

    def get_TFIDF(self, terms):
        pass

    # implementar y llamar luego de crear el índice, para llenar
    # el vector self._norms
    def get_norms(self):
        # usar el compute_norm
        pass

    # vector de tipo get_TFIDF
    def compute_norm(self, vector):
        v = np.array(vector)
        return np.linalg.norm(v)

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