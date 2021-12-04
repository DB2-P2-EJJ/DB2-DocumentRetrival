import os
import unittest
import shutil
from inverted_index import MailInvertedIndex
from pathlib import Path


class InvertedIndexTest(unittest.TestCase):
    def test_constructor(self):
        mii = MailInvertedIndex()
        self.assertTrue(mii.is_sorted())
        shutil.rmtree(os.getcwd() / Path('email.mii'))

    def test_query(self):
        mii = MailInvertedIndex()
        q = "line"
        print("query 1: ", mii.query(q, 15))
        q2 = "the best hunting"
        print("query 2: ", mii.query(q2, 15))
        # shutil.rmtree(os.getcwd() / Path('email.mii'))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
