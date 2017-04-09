from recsys.util.common import app_init

import argparse
import movie_lens_data as mld
import os
import recsys.util.feature_helper as fh
import tensorflow as tf
import unittest

class TestLoadData(unittest.TestCase):
    def test_load_item_profile(self):
        TEST_SRCDIR = os.environ["TEST_SRCDIR"]
        movie_file = os.path.join(os.environ["TEST_SRCDIR"],"org_tensorflow/recsys/application/movie_lens/test_data/movie_sample.csv")
        item_dict = mld.load_item_profile(movie_file)
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
        rating_file = os.path.join(os.environ["TEST_SRCDIR"],"org_tensorflow/recsys/application/movie_lens/test_data/ratings_sample.csv")
        tf_records_file = os.path.join(os.environ["TEST_TMPDIR"],"ratings_sample.tfrecord")
        mld.convert_to_tf_records(rating_file, tf_records_file)
        # Reads back to ensure the correctness
        record_iterator = tf.python_io.tf_record_iterator(path = tf_records_file)
        for str_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(str_record)
            self.assertEqual(fh.get_string_feature(example, 'user_id'),'1')
            self.assertEqual(fh.get_string_feature(example, 'item_id'),'31')
            self.assertAlmostEqual(round(fh.get_float_feature(example, 'rating'),2),2.5)


if __name__ == '__main__':
    prog_name, unparsed = app_init()
    unittest.main(argv = [prog_name] + unparsed)
