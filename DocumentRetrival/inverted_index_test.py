import os
import unittest
import shutil
from inverted_index import MailInvertedIndex
from pathlib import Path


class InvertedIndexTest(unittest.TestCase):
    def test_constructor(self):
        mii = MailInvertedIndex()
        self.assertTrue(mii.is_sorted('index', mii.n_index_block))
        self.assertTrue(mii.is_sorted('length', mii.n_length_block))
        # shutil.rmtree(os.getcwd() / Path('email.mii'))

    def test_query(self):
        mii = MailInvertedIndex()
        q = "line"
        print("query 1: ", mii.query(q, 15))
        q = "the best hunting"
        print("query 2: ", mii.query(q, 15))
        self.assertTrue(True)
        shutil.rmtree(os.getcwd() / Path('email.mii'))


if __name__ == '__main__':
    unittest.main()
