from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf

import ml.common.flags as my_flags
import ml.module.proto.feature_column_schema_pb2 as fcs_pb2
import ml.module.feature_column_factory as fcf

FLAGS = my_flags.FLAGS


class FeatureColumnFactoryTest(tf.test.TestCase):

  def setUp(self):
    self.schema_pbtxt = os.path.join(
        FLAGS.test_srcdir, "ml/module/testing_data/feature_column_schema.pbtxt")
    self.int_fixed = tf.contrib.layers.real_valued_column(
        "int_fixed", dimension=10, dtype=tf.int64)
    self.label = tf.contrib.layers.real_valued_column(
        fcf.FEATURE_LABEL_NAME, dimension=1, dtype=tf.int64)
    self.float_fixed = tf.contrib.layers.real_valued_column(
        "float_fixed", dimension=20, dtype=tf.float32)
    self.integerized_sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
        "integerized_sparse", bucket_size=10, combiner="sum", dtype=tf.int32)
    self.hash_bucket_sparse = tf.contrib.layers.sparse_column_with_hash_bucket(
        "hash_bucket_sparse", hash_bucket_size=5000, combiner="mean", dtype=tf.int64)

  def testNotIncludeTarget(self):
    self.feature_columns = fcf.feature_columns_from_file(
        self.schema_pbtxt, include_target=False)
    self.assertItemsEqual(self.feature_columns, set(
        [self.int_fixed, self.float_fixed, self.integerized_sparse, self.hash_bucket_sparse]))

  def testIncludeTarget(self):
    self.feature_columns = fcf.feature_columns_from_file(
        self.schema_pbtxt, include_target=True)
    self.assertItemsEqual(self.feature_columns, set(
        [self.int_fixed, self.label, self.float_fixed, self.integerized_sparse, self.hash_bucket_sparse]))

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
