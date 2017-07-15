import logging
import math
import os
import recsys.util.common as ruc
import recsys.util.dataset_helper as rud
import recsys.util.feature_helper as ruf
import tensorflow as tf
import unittest


def _create_tmp_file(file_name):
  return os.path.join(os.environ['TEST_TMPDIR'], file_name)


class DataHelperTest(unittest.TestCase):

  def setUp(self):
    self.tf_record_file = _create_tmp_file("examples.tfrecords")
    self.num_records = 1000
    with tf.python_io.TFRecordWriter(self.tf_record_file) as writer:
      for i in range(self.num_records):
        writer.write(ruf.example_from_feature_dict({
            'x': ruf.int64_feature(i),
            'y': ruf.string_feature('pos')
        }).SerializeToString())

  def test_number_of_tf_records(self):
    self.assertEqual(
        self.num_records, rud.number_of_tf_records(self.tf_record_file))

  def test_split_train_test(self):
    train_ratio = 0.7
    train_file, test_file = rud.split_train_test(
        self.tf_record_file, train_ratio)
    num_train_records = rud.number_of_tf_records(train_file)
    num_test_records = rud.number_of_tf_records(test_file)
    logging.info('train records ratio:{}'.format(
        num_train_records / self.num_records))
    # Most times the train records ratio should be within
    # 0.7+/-1.96*sqrt(0.7*0.3/1000)
    self.assertGreaterEqual(num_train_records / self.num_records, 0.67)
    self.assertLessEqual(num_train_records / self.num_records, 0.73)

  def test_cv_split(self):
    cv_folds = 5
    cv_files = rud.generate_cv_partitions(self.tf_record_file, cv_folds)
    error_width = math.sqrt((1 / cv_folds) * (1 - 1 / cv_folds) / 1000)
    avg_ratio = 1 / cv_folds
    error_lb = avg_ratio - 2 * error_width
    error_ub = avg_ratio + 2 * error_width
    logging.info(
        'error lower bound: {}, upper bound: {}'.format(error_lb, error_ub))
    for f in cv_files:
      n = rud.number_of_tf_records(f)
      self.assertGreaterEqual(n / self.num_records, error_lb)
      self.assertLessEqual(n / self.num_records, error_ub)
      logging.info('cv file:{} with {} records'.format(
          f, rud.number_of_tf_records(f)))

if __name__ == '__main__':
  prog_name, unparsed = ruc.app_init()
  unittest.main(argv=[prog_name] + unparsed)
