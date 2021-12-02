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
        mii.query(q, 15)


if __name__ == '__main__':
    unittest.main()
