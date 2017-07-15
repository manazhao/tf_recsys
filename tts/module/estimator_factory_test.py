from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import tensorflow as tf

import tts.common.flags as my_flags
import tts.module.proto.estimator_config_pb2 as est_pb2
import tts.module.estimator_factory as est_factory

FLAGS = my_flags.FLAGS


class EstimatorFactoryTest(tf.test.TestCase):

  def setUp(self):
    self.config_file = os.path.join(
        FLAGS.test_srcdir, "tts/module/testing_data/estimator_config.pbtext")
    self.estimator_config = est_factory.estimator_config_from_file(
        self.config_file)
    # self.estimator_config = est_pb2.EstimatorConfig()
    # self.estimator_config.estimator_type = (
    #     est_pb2.EstimatorConfig.ESTIMATOR_TYPE_DNN_CLASSIFIER)
    # dnn_config = self.estimator_config.dnn_classifier_config
    # dnn_config.common.num_classes = 2
    # dnn_config.common.dropout = 0
    # dnn_config.common.learning_rate = 0.1
    # dnn_config.common.weight_column_name = ""

    self.run_config = tf.contrib.learn.RunConfig()
    self.feature_columns = set(
        {tf.contrib.layers.real_valued_column(column_name="x", dimension=1, dtype=tf.float32)})

  def test_create_estimator(self):
    with tf.Graph().as_default(), tf.Session():
      dnn_classifier = est_factory.create_estimator(
          run_config=self.run_config, estimator_config=self.estimator_config, feature_columns=self.feature_columns)
      self.assertIsNotNone(dnn_classifier)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
