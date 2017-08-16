from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.core.framework import types_pb2

import ml.module.proto.feature_column_schema_pb2 as fcs_pb2

FEATURE_LABEL_NAME = "label"


def combiner_type_to_string(combiner):
  if combiner == fcs_pb2.COMBINER_TYPE_MEAN:
    return "mean"
  elif combiner == fcs_pb2.COMBINER_TYPE_SQRT:
    return "sqrt"
  elif combiner == fcs_pb2.COMBINER_TYPE_SUM:
    return "sum"
  else:
    raise ValueError("unsupported combiner type: %s" % (combiner))


class FeatureColumnStrategy(object):

  def __init__(self, schema_file):
    self._schema_file = schema_file
    self._column_lu = {}
    self._config_lu = {}
    self._visited_configs = set()
    self._included_columns = set()
    self._schema_proto = None
    with tf.gfile.GFile(self._schema_file, "r") as f:
      self._schema_proto = fcs_pb2.FeatureColumnConfigSchema()
      text_format.Parse(f.read(), self._schema_proto)
      self._config_lu = dict((c.name, c)
                             for c in self._schema_proto.column_config)

  def _get_dependent_configs(self, config):
    return [self._config_lu[name] for name in config.dependency]

  def _get_dependent_columns(self, config):
    return [self._column_lu[name] for name in config.dependency]

  def _add_feature_columns_for_config(self, config):
    if config.name in self._visited_configs:
      return
    for dependent_config in self._get_dependent_configs(config):
      self._add_feature_columns_for_config(dependent_config)

    which_column = config.WhichOneof("specific_column")
    columns = set()
    if which_column == "real_valued":
      real_valued = config.real_valued
      column = tf.contrib.layers.real_valued_column(
          config.name,
          dimension=real_valued.dimension,
          dtype=tf.DType(real_valued.dtype))
      columns.add(column)
      if config.include:
        self._included_columns.add(column.name)
    elif which_column == "bucketized":
      raise NotImplementedError(
          "no implementation for %s column" % which_column)
    elif which_column == "one_hot":
      sparse_column = self._get_dependent_columns(config).pop()
      column = tf.contrib.layers.one_hot_column(sparse_column)
      columns.add(column)
      if config.include:
        self._included_columns.add(column.name)
    elif which_column == "hash_bucket":
      hash_bucket = config.hash_bucket
      column = tf.contrib.layers.sparse_column_with_hash_bucket(
          config.name,
          hash_bucket_size=hash_bucket.hash_bucket_size,
          combiner=combiner_type_to_string(hash_bucket.combiner),
          dtype=tf.DType(hash_bucket.dtype))
      columns.add(column)
      if config.include:
        self._included_columns.add(column.name)
    elif which_column == "integerized":
      integerized = config.integerized
      column = tf.contrib.layers.sparse_column_with_integerized_feature(
          config.name,
          bucket_size=integerized.bucket_size,
          dtype=tf.DType(integerized.dtype))
      columns.add(column)
      if config.include:
        self._included_columns.add(column.name)
    elif which_column == "embedding":
      sparse_column = self._get_dependent_columns(config).pop()
      embedding = config.embedding
      column = tf.contrib.layers.embedding_column(
          sparse_column,
          dimension=embedding.dimension,
          combiner=combiner_type_to_string(embedding.combiner))
      columns.add(column)
      if config.include:
        self._included_columns.add(column.name)
    elif which_column == "shared_embedding":
      sparse_columns = self._get_dependent_columns(config)
      shared = config.shared_embedding
      shared_columns = tf.contrib.layers.contrib.shared_embedding_columns(
          sparse_columns,
          dimension=shared.dimension,
          combiner=combiner_type_to_string(shared.combiner),
          dtype=tf.DType(shared.dtype))
      columns.add(shared_columns)
      if config.include:
        self._included_columns.add([c.name for c in shared_columns])

    self._visited_configs.add(config.name)
    self._column_lu.update(dict((c.name, c) for c in columns))

  def get_feature_columns(self, include_target):
    for config in self._schema_proto.column_config:
      self._add_feature_columns_for_config(config)
    skipped_columns = set([] if include_target else [FEATURE_LABEL_NAME])
    return dict(
        (c.name, c) for c in self._column_lu.itervalues()
        if c.name in self._included_columns and c.name not in skipped_columns)
