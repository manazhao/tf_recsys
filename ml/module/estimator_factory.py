from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from google.protobuf import text_format
import tensorflow as tf
import ml.module.proto.estimator_config_pb2 as est_pb2


def estimator_config_from_file(config_file):
  """Loads `EstimatorConfig` from text proto file.

   Reads in the configuration file and parse it as an object.

   Args:
    config_file: (`EstimatorConfig`) text proto file.

   Returns:
    `EstimatorConfig` object.
  """
  with tf.gfile.GFile(config_file, "r") as f:
    estimator_config = est_pb2.EstimatorConfig()
    text_format.Parse(f.read(), estimator_config)
    return estimator_config


def create_estimator(run_config, estimator_config, feature_columns):
  """Creates an estimator based on the configuration.

    Args:
      run_config: (`tf.contrib.learn.RunConfig`) object.
      estimator_config: (`est_pb2.EstimatorConfig`) estimator configuration.
      feature_columns: a set of `FeatureColumn` objects.

    Raises:
      NotImplementedError: unimplemented estimator type.

    Returns:
          An tf.learn.Estimator object.
  """
  if (estimator_config.estimator_type ==
      est_pb2.EstimatorConfig.ESTIMATOR_TYPE_DNN_CLASSIFIER):
    estimator_config = estimator_config.dnn_classifier_config
    return tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        config=run_config,
        hidden_units=estimator_config.hidden_units,
        n_classes=estimator_config.common.num_classes,
        weight_column_name=estimator_config.common.weight_column_name,
        optimizer=tf.train.AdagradOptimizer(
            estimator_config.common.learning_rate))
