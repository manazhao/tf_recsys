from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf

import ml.common.flags as my_flags
import ml.module.feature_column_factory as fcf
import ml.module.proto.feature_column_schema_pb2 as fcs_pb2

FLAGS = my_flags.FLAGS


class FeatureColumnFactoryTest(tf.test.TestCase):

  def setUp(self):
    self._schema_pbtxt = os.path.join(
        FLAGS.test_srcdir, "ml/module/testing_data/feature_column_schema.pbtxt")
    self._factory = fcf.FeatureColumnStrategy(self._schema_pbtxt)
    self._int32 = tf.contrib.layers.real_valued_column(
        "int32", dimension=10, dtype=tf.int32)
    self._label = tf.contrib.layers.real_valued_column(
        fcf.FEATURE_LABEL_NAME, dimension=1, dtype=tf.int64)
    self._float = tf.contrib.layers.real_valued_column(
        "float", dimension=20, dtype=tf.float32)
    self._integerized = (
        tf.contrib.layers.sparse_column_with_integerized_feature(
            "integerized", bucket_size=10, combiner="sum", dtype=tf.int32))
    self._hash_bucket = tf.contrib.layers.sparse_column_with_hash_bucket(
        "hash_bucket", hash_bucket_size=5000, combiner="mean", dtype=tf.int64)
    self._embedding = tf.contrib.layers.embedding_column(
        sparse_id_column=self._hash_bucket, dimension=50, combiner="sqrt")

  def _assertEmbeddingColumnEqual(self, column1, column2):
    column1 = column1._replace(initializer=None)
    column2 = column2._replace(initializer=None)
    self.assertEqual(column1, column2)

  def testNotIncludeTarget(self):
    feature_columns_dict = self._factory.get_feature_columns(
        include_target=False)

    self.assertItemsEqual([
        "int32", "float", "integerized", "hash_bucket", "hash_bucket_embedding"
    ], feature_columns_dict.keys())
    self.assertEqual(feature_columns_dict["int32"], self._int32)
    self.assertEqual(feature_columns_dict["float"], self._float)
    self.assertEqual(feature_columns_dict["integerized"], self._integerized)
    self.assertEqual(feature_columns_dict["hash_bucket"], self._hash_bucket)
    self._assertEmbeddingColumnEqual(
        feature_columns_dict["hash_bucket_embedding"], self._embedding)

  def testIncludeTarget(self):
    feature_columns_dict = self._factory.get_feature_columns(
        include_target=True)

    self.assertItemsEqual([
        "int32", "float", "integerized", "hash_bucket", "label",
        "hash_bucket_embedding"
    ], feature_columns_dict.keys())
    self.assertEqual(feature_columns_dict["int32"], self._int32)
    self.assertEqual(feature_columns_dict["label"], self._label)
    self.assertEqual(feature_columns_dict["float"], self._float)
    self.assertEqual(feature_columns_dict["integerized"], self._integerized)
    self.assertEqual(feature_columns_dict["hash_bucket"], self._hash_bucket)
    self._assertEmbeddingColumnEqual(
        feature_columns_dict["hash_bucket_embedding"], self._embedding)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
