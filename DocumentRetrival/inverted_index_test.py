import os
import unittest
import shutil
from inverted_index import MailInvertedIndex
from pathlib import Path


class InvertedIndexTest(unittest.TestCase):
    def test_constructor(self):
        mii = MailInvertedIndex()
        self.assertIsInstance(mii, MailInvertedIndex)
        # shutil.rmtree(os.getcwd() / Path('email.mii'))


if __name__ == '__main__':
    unittest.main()
