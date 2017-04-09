from common import app_init

import argparse
import feature_helper as fh
import logging
import os
import tensorflow as tf
import tensorflow.contrib.recsys.util.proto.config_pb2 as config
import unittest


def _generate_example(int_x, float_y):
    example = fh.example_from_feature_dict({
            'x' : fh.int64_feature(int_x),
            'y' : fh.float_feature(float_y)
        })
    return example

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
    def test_fetch_and_process_features(self):
        # Generates some tf.Examples
        example1 = _generate_example(1,3)
        example2 = _generate_example(2,4)
        tf_records_file = os.path.join(os.environ['TEST_TMPDIR'],"data.tfrecords")
        with tf.python_io.TFRecordWriter(tf_records_file) as writer:
            writer.write(example1.SerializeToString())
            writer.write(example2.SerializeToString())
        input_config = config.FeatureInputConfig()
        input_config.batch_size = 1
        input_config.num_batches = 2
        input_config.capacity = 1
        input_config.num_epochs = 1
        input_config.num_threads = 1
        input_config.min_after_dequeue = 1

        def _consume_batched_features(features, labels):
            # logging.info(features)
            # logging.info(labels)
            logging.info('handling features and labels')

        fh.fetch_and_process_features([tf_records_file],
                x_feature_spec = {'x' : tf.FixedLenFeature([],tf.int64)},
                y_feature_spec = {'y' : tf.FixedLenFeature([], tf.float32)},
                input_config = input_config,
                consume_batch_fn = _consume_batched_features
            )

if __name__ == '__main__':
    prog_name, unparsed = app_init()
    unittest.main(argv = [prog_name] + unparsed)
