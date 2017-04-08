import load_data as ld
from common import app_init

import argparse
import os
import unittest

class TestLoadData(unittest.TestCase):
    def test_load_item_profile(self):
        TEST_SRCDIR = os.environ["TEST_SRCDIR"]
        movie_file = os.path.join(os.environ["TEST_SRCDIR"],"org_tensorflow/tensorflow/contrib/recsys/util/test_data/movie_sample.csv")
        item_dict = ld.load_item_profile(movie_file)
        expected_dict = {
        	1 : {"title" : "toy story", "year" : "1995", "genres" :["adventure","animation","children","comedy","fantasy"]},
        	2 : {"title" : "jumanji", "year" : "1995", "genres" : ["adventure", "children", "fantasy"]}
        }
        self.assertSequenceEqual(item_dict.keys(), expected_dict.keys())
        for k in item_dict.keys():
        	self.assertEqual(item_dict[k]["year"],expected_dict[k]["year"])
        	self.assertEqual(item_dict[k]["title"],expected_dict[k]["title"])
        	self.assertListEqual(item_dict[k]["genres"],expected_dict[k]["genres"])
    def test_load_ratings(self):
        TEST_SRCDIR = os.environ["TEST_SRCDIR"]
        rating_file = os.path.join(os.environ["TEST_SRCDIR"],"org_tensorflow/tensorflow/contrib/recsys/util/test_data/ratings_sample.csv")
        examples = ld.convert_to_example(rating_file)
        logging.info(examples)

if __name__ == '__main__':
    prog_name, unparsed = app_init()
    unittest.main(argv = [prog_name] + unparsed)
