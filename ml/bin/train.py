#! /usr/bin/env python
"""Main script to run training and evaluation of models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow import gfile

import ml.module.estimator_factory as ef
import ml.module.feature_column_factory as fcf

tf.flags.DEFINE_integer("batch_size", 16,
                        """Batch size used for training and evaluation.""")
tf.flags.DEFINE_string("output_dir", None,
                       """The directory to write model checkpoints and summaries
                       to. If None, a local temporary directory is created.""")

tf.flags.DEFINE_string("estimator_config_file", None,
                       "Estimator configuration pbtxt file")
tf.flags.DEFINE_string("feature_column_schema_file", None,
                       "`FeatureColumnConfigSchema` pbtxt file")
# Training data.
tf.flags.DEFINE_string("training_data", None,
                       "File patterns for training data.")
tf.flags.DEFINE_string("testing_data", None, "File patterns for testing data.")

# Training parameters
tf.flags.DEFINE_string("schedule", "continuous_train_and_eval",
                       """Estimator function to call, defaults to
                       continuous_train_and_eval for local run""")
tf.flags.DEFINE_integer("train_steps", None,
                        """Maximum number of training steps to run.
                         If None, train forever.""")
tf.flags.DEFINE_integer("eval_steps", None,
                        "Number of evaluation steps per checkpoint.")
tf.flags.DEFINE_integer("eval_every_n_steps", 1000,
                        "Run evaluation on validation data every N steps.")

# RunConfig Flags
tf.flags.DEFINE_integer("tf_random_seed", None,
                        """Random seed for TensorFlow initializers. Setting
                        this value allows consistency between reruns.""")
tf.flags.DEFINE_integer("save_checkpoints_secs", None,
                        """Save checkpoints every this many seconds.
                        Can not be specified with save_checkpoints_steps.""")
tf.flags.DEFINE_integer("save_checkpoints_steps", None,
                        """Save checkpoints every this many steps.
                        Can not be specified with save_checkpoints_secs.""")
tf.flags.DEFINE_integer("keep_checkpoint_max", 5,
                        """Maximum number of recent checkpoint files to keep.
                        As new files are created, older files are deleted.
                        If None or 0, all checkpoint files are kept.""")
tf.flags.DEFINE_integer("keep_checkpoint_every_n_hours", 4,
                        """In addition to keeping the most recent checkpoint
                        files, keep one checkpoint file for every N hours of
                        training.""")
tf.flags.DEFINE_float("gpu_memory_fraction", 1.0,
                      """Fraction of GPU memory used by the process on
                      each GPU uniformly on the same machine.""")
tf.flags.DEFINE_boolean("gpu_allow_growth", False,
                        """Allow GPU memory allocation to grow
                        dynamically.""")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        """Log the op placement to devices""")

FLAGS = tf.flags.FLAGS


def _get_estimator(run_config, feature_columns):
  """Creates an `Estimator`.

  Args:
    config: (`RunConfig`) Configuration for learn runner.
    feature_columns: a set of `tf._FeatureColumn`s.

  Returns:
    a `tf.Estimator` object.
  """
  estimator_config = ef.estimator_config_file(FLAGS.estimator_config_file)
  estimator = ef.create_estimator(run_config, estimator_config, feature_columns)


def _get_feature_columns(include_target):
  feature_column_strategy = fcf.FeatureColumnStrategy(
      FLAGS.feature_column_schema_file)
  feature_columns_dict = feature_column_strategy.get_feature_columns(
      include_target=include_target)
  return feature_columns_dict.values()


def _get_input_fn(mode, batch_size):

  def _input_fn():
    include_target = (mode != tf.contrib.learn.ModeKeys.INFER)
    file_patterns = FLAGS.training_data.split(",")
    randomize_input = True if mode == tf.contrib.learn.ModeKeys.TRAIN else False
    feature_spec = (tf.contrib.layers.create_feature_spec_for_parsing(
        feature_columns=_get_feature_columns(include_target)))
    feature_map = tf.contrib.learn.io.read_batch_features(
        file_pattern=file_patterns,
        batch_size=batch_size,
        reader_num_threads=1,
        features=feature_spec,
        randomize_input=randomize_input)
    target = None
    if mode != tf.contrib.learn.ModeKeys.INFER:
      target = feature_map.pop(fcf.FEATURE_LABEL_NAME)
    return feature_map, target

  return _input_fn


def _create_experiment(run_config, hparams):
  """
  Creates a new Experiment instance.

  Args:
    run_config: (`tf.RunConfig`) configuration for the experiment.
    hparams: (`tf.HParams`) model hyperparameters.
  """
  _ = hparams
  feature_column_strategy = fcf.FeatureColumnStrategy(
      FLAGS.feature_column_schema_file)
  feature_columns_dict = feature_column_strategy.get_feature_columns(
      include_target=True)
  estimator = _get_estimator(run_config, feature_columns_dict.values())

  train_input_fn = _get_input_fn(tf.contrib.learn.ModeKeys.TRAIN,
                                 FLAGS.batch_size)
  eval_input_fn = _get_input_fn(tf.contrib.learn.ModeKeys.EVAL,
                                FLAGS.batch_size)
  serving_input_fn = tf.contrib.layers.build_parsing_serving_input_fn(
      feature_spec)
  export_strategy = tf.contrib.learn.make_export_strategy(
      serving_input_fn, exports_to_keep=None)

  experiment = tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      min_eval_frequency=FLAGS.eval_every_n_steps,
      train_steps=FLAGS.train_steps,
      eval_steps=FLAGS.eval_steps,
      export_strategies=[export_strategy])

  return experiment


def main(_argv):
  if not FLAGS.output_dir:
    FLAGS.output_dir = tempfile.mkdtemp()
  run_config = tf.contrib.learn.RunConfig(model_dir=FLAGS.output_dir)
  learn_runner.run(
      experiment_fn=_create_experiment, run_config=run_config, hparams=None)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
