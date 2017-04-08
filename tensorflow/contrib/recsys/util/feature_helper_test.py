from common import app_init

import argparse
import feature_helper as fh
import logging
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
    	example = fh.example_from_feature_dict({
    			"int_val" : fh.int64_feature(int_val),
    			"int_list" : fh.int64_list_feature(int_list),
    			"float_val" : fh.float_feature(float_val),
    			"float_list" : fh.float_list_feature(float_list),
    			"str_val" : fh.string_feature(str_val),
    			"str_list" : fh.string_list_feature(str_list)
    		})
    	example_str = example.SerializeToString()
    	parsed_example = tf.train.Example()
    	parsed_example.ParseFromString(example_str)
    	self.assertEqual(fh.get_int64_feature(parsed_example,'int_val'), 0)
    	self.assertListEqual(fh.get_int64_list_feature(parsed_example, 'int_list'), int_list)
    	self.assertAlmostEqual(round(fh.get_float_feature(parsed_example,'float_val'),2), float_val)
    	self.assertListEqual([round(v,2) for v in fh.get_float_list_feature(parsed_example,'float_list')], float_list)
    	self.assertEqual(fh.get_string_feature(parsed_example,'str_val'), str_val)
    	self.assertListEqual(fh.get_string_list_feature(parsed_example,'str_list'), str_list)

if __name__ == '__main__':
    prog_name, unparsed = app_init()
    unittest.main(argv = [prog_name] + unparsed)
