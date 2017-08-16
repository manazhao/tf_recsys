from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import tensorflow as tf

import ml.common.flags as my_flags
import ml.module.estimator_factory as est_factory
import ml.module.proto.estimator_config_pb2 as est_pb2

FLAGS = my_flags.FLAGS


class EstimatorFactoryTest(tf.test.TestCase):

  def setUp(self):
    self.config_file = os.path.join(
        FLAGS.test_srcdir, "ml/module/testing_data/estimator_config.pbtxt")
    self.estimator_config = est_factory.estimator_config_from_file(
        self.config_file)

    self.run_config = tf.contrib.learn.RunConfig()
    self.feature_columns = set({
        tf.contrib.layers.real_valued_column(
            column_name="x", dimension=1, dtype=tf.float32)
    })

  def test_create_estimator(self):
    with tf.Graph().as_default(), tf.Session():
      dnn_classifier = est_factory.create_estimator(
          run_config=self.run_config,
          estimator_config=self.estimator_config,
          feature_columns=self.feature_columns)
      self.assertIsNotNone(dnn_classifier)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
