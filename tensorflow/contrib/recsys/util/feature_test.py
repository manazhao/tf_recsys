from common import app_init

import argparse
import feature
import tensorflow as tf
import unittest

class TestFeature(unittest.TestCase):
    def test_construct_example(self):
    	int_val = 0
    	int_list = [1, 2]
    	float_val = 3.14
    	float_list = [4.1, 5.4]
    	str_val = "hello"
    	str_list = ["tensor", "flow"]
    	example = feature.example_from_feature_dict({
    			"int_val" : feature.int64_feature(int_val),
    			"int_list" : feature.int64_list_feature(int_list),
    			"float_val" : feature.float_feature(float_val),
    			"float_list" : feature.float_list_feature(float_list),
    			"str_val" : feature.string_feature(str_val),
    			"str_list" : feature.string_list_feature(str_list)
    		})
    	serialized_example = example.SerializeToString()
    	parsed_features = tf.parse_single_example(
    		serialized_example,
    		features = {
    			"int_val" : tf.FixedLenFeature([], tf.int64),
    			"int_list" : tf.FixedLenFeature([], tf.int64),
    			"float_val" : tf.FixedLenFeature([], tf.float32),
    			"float_list" : tf.FixedLenFeature([], tf.float32),
    			"str_val" : tf.VarLenFeature(tf.string),
    			"str_list" : tf.VarLenFeature(tf.string)
    		}
    	)
    	self.assertEqual(tf.cast(parsed_features["int_val"], tf.int64), int_val)
    	print(example_parsed)


if __name__ == '__main__':
    prog_name, unparsed = app_init()
    unittest.main(argv = [prog_name] + unparsed)
