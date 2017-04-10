import logging
import os
import recsys.util.common as ruc
import recsys.util.dataset_helper as rud
import recsys.util.feature_helper as ruf
import tensorflow as tf
import unittest

def _create_tmp_file(file_name):
	return os.path.join(os.environ['TEST_TMPDIR'],file_name)

class DataHelperTest(unittest.TestCase):
	def setUp(self):
		self.tf_record_file = _create_tmp_file("examples.tfrecords")
		self.num_records = 10000
		with tf.python_io.TFRecordWriter(self.tf_record_file) as writer:
			for i in range(self.num_records):
				writer.write(ruf.example_from_feature_dict({
					'x': ruf.int64_feature(i),
					'y' : ruf.string_feature('pos')
				}).SerializeToString())

	def test_number_of_tf_records(self):
		self.assertEqual(self.num_records, rud.number_of_tf_records(self.tf_record_file))

	def test_split_train_test(self):
		train_ratio = 0.7
		train_file, test_file = rud.split_train_test(self.tf_record_file, train_ratio)
		num_train_records = rud.number_of_tf_records(train_file)
		num_test_records = rud.number_of_tf_records(test_file)
		self.assertAlmostEqual(num_train_records/num_test_records, train_ratio/(1 - train_ratio), places = 1)


if __name__ == '__main__':
    prog_name, unparsed = ruc.app_init()
    unittest.main(argv = [prog_name] + unparsed)
