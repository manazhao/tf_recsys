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

tf.flags.DEFINE_integer("batch_size", 16,
                        """Batch size used for training and evaluation.""")
tf.flags.DEFINE_string("output_dir", None,
                       """The directory to write model checkpoints and summaries
                       to. If None, a local temporary directory is created.""")

# Training parameters
tf.flags.DEFINE_string("schedule", "continuous_train_and_eval",
                       """Estimator function to call, defaults to
                       continuous_train_and_eval for local run""")
tf.flags.DEFINE_integer("train_steps", None,
                        """Maximum number of training steps to run.
                         If None, train forever.""")
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

def _get_estimator(run_config, hparams, estimator_config):
  """Creates an `Estimator`.

  Args:
    config: (`RunConfig`) Configuration for learn runner.
    hparams: (`HParams`) Model hyperparameter specification.
    estimator
  """

def create_experiment(run_config, hparams):
  """
  Creates a new Experiment instance.

  Args:
    run_config: (RunConfig) configuration for the experiment.
    hparams: (HParams) estimator hyperparameters.
  """

  config = run_config.RunConfig(
      tf_random_seed=FLAGS.tf_random_seed,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      gpu_memory_fraction=FLAGS.gpu_memory_fraction)
  config.tf_config.gpu_options.allow_growth = FLAGS.gpu_allow_growth
  config.tf_config.log_device_placement = FLAGS.log_device_placement


  estimator = tf.contrib.learn.Estimator(
      model_fn=model_fn,
      model_dir=output_dir,
      config=config,
      params=FLAGS.model_params)

  # Create hooks
  train_hooks = []
  for dict_ in FLAGS.hooks:
    hook = _create_from_dict(
        dict_, hooks,
        model_dir=estimator.model_dir,
        run_config=config)
    train_hooks.append(hook)

  # Create metrics
  eval_metrics = {}
  for dict_ in FLAGS.metrics:
    metric = _create_from_dict(dict_, metric_specs)
    eval_metrics[metric.name] = metric

  experiment = PatchedExperiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      min_eval_frequency=FLAGS.eval_every_n_steps,
      train_steps=FLAGS.train_steps,
      eval_steps=None,
      eval_metrics=eval_metrics,
      train_monitors=train_hooks)

  return experiment


def main(_argv):
  """The entrypoint for the script"""

  # Parse YAML FLAGS
  FLAGS.hooks = _maybe_load_yaml(FLAGS.hooks)
  FLAGS.metrics = _maybe_load_yaml(FLAGS.metrics)
  FLAGS.model_params = _maybe_load_yaml(FLAGS.model_params)
  FLAGS.input_pipeline_train = _maybe_load_yaml(FLAGS.input_pipeline_train)
  FLAGS.input_pipeline_dev = _maybe_load_yaml(FLAGS.input_pipeline_dev)

  # Load flags from config file
  final_config = {}
  if FLAGS.config_paths:
    for config_path in FLAGS.config_paths.split(","):
      config_path = config_path.strip()
      if not config_path:
        continue
      config_path = os.path.abspath(config_path)
      tf.logging.info("Loading config from %s", config_path)
      with gfile.GFile(config_path.strip()) as config_file:
        config_flags = yaml.load(config_file)
        final_config = _deep_merge_dict(final_config, config_flags)

  tf.logging.info("Final Config:\n%s", yaml.dump(final_config))

  # Merge flags with config values
  for flag_key, flag_value in final_config.items():
    if hasattr(FLAGS, flag_key) and isinstance(getattr(FLAGS, flag_key), dict):
      merged_value = _deep_merge_dict(flag_value, getattr(FLAGS, flag_key))
      setattr(FLAGS, flag_key, merged_value)
    elif hasattr(FLAGS, flag_key):
      setattr(FLAGS, flag_key, flag_value)
    else:
      tf.logging.warning("Ignoring config flag: %s", flag_key)

  if FLAGS.save_checkpoints_secs is None \
    and FLAGS.save_checkpoints_steps is None:
    FLAGS.save_checkpoints_secs = 600
    tf.logging.info("Setting save_checkpoints_secs to %d",
                    FLAGS.save_checkpoints_secs)

  if not FLAGS.output_dir:
    FLAGS.output_dir = tempfile.mkdtemp()

  if not FLAGS.input_pipeline_train:
    raise ValueError("You must specify input_pipeline_train")

  if not FLAGS.input_pipeline_dev:
    raise ValueError("You must specify input_pipeline_dev")

  learn_runner.run(
      experiment_fn=create_experiment,
      output_dir=FLAGS.output_dir,
      schedule=FLAGS.schedule)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
