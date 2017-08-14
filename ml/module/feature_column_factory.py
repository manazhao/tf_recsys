from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

from google.protobuf import text_format
import tensorflow as tf

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


def data_type_to_string(data_type):
  if data_type == fcs_pb2.DATA_TYPE_INT32:
    return tf.int32
  elif data_type == fcs_pb2.DATA_TYPE_INT64:
    return tf.int64
  elif data_type == fcs_pb2.DATA_TYPE_FLOAT32:
    return tf.float32
  elif data_type == fcs_pb2.DATA_TYPE_FLOAT64:
    return tf.float64
  elif data_type == fcs_pb2.DATA_TYPE_STRING:
    return tf.string
  else:
    raise ValueError("unsupported data type: %s" % (data_type))


def feature_column_schema_from_config(schema_pbtxt):
  """Loads `FeatureColumnSchema` from text proto file.

  Reads in the schema file and parse it as an object.
  Args:
    schema_pbtxt: (`FeatureColumnSchema`) text proto file.

  Returns:
    `FeatureColumnSchema` object.
  """
  with tf.gfile.GFile(schema_pbtxt, "r") as f:
    schema = fcs_pb2.FeatureColumnSchema()
    text_format.Parse(f.read(), schema)
    return schema


def feature_columns_from_file(schema_pbtxt, include_target):
  """Creates feature columns by loading `FeatureColumnSchema` from config file.

  Args:
    schema_pbtxt: file path of `FeatureColumnSchema` protobuf text.
    include_target: a boolean indicating whether return feature column for the target feature.

  Raises:
    AssertionError: when oneof field is not set.

  Returns:
    a set of `_FeatureColumn`.
  """
  schema = feature_column_schema_from_config(schema_pbtxt)
  # Key is column name and value is *Column defined in
  # feature_column_schema.proto.
  config_lu = dict()
  # Key is column name and value is `_FeatureColumn`.
  column_lu = dict()
  # Key is column name and value is `bool` indicating whether the column
  # should be included in the result.
  retain_lu = dict()
  for column in schema.basic_columns:
    name = column.name
    retain_lu[name] = column.retain
    # Skips label feautre column if include_target is false.
    if name == FEATURE_LABEL_NAME and not include_target:
      continue
    which_column = column.WhichOneof("column")
    assert which_column is not None
    # Gets `_FeatureColumn` for the oneof fields of `BasicFeatureColumn`.
    if which_column == "int_fixed":
      column_lu[name] = _int_fixed_column_handler(column)
      config_lu[name] = column.int_fixed
    elif which_column == "int_var":
      column_lu[name] = _int_var_column_handler(column)
      config_lu[name] = column.int_var
    elif which_column == "float_fixed":
      column_lu[name] = _float_fixed_column_handler(column)
      config_lu[name] = column.float_fixed
    elif which_column == "float_var":
      column_lu[name] = _float_var_column_handler(column)
      config_lu[name] = column.float_var
    elif which_column == "integerized_sparse":
      column_lu[name] = _integerized_sparse_column_handler(column)
      config_lu[name] = column.integerized_sparse
    elif which_column == "hash_bucket_sparse":
      column_lu[name] = _hash_bucket_sparse_column_handler(column)
      config_lu[name] = column.hash_bucket_sparse
    else:
      raise ValueError("unimplemented handler for column: %s" % (which_column))

  # Gets config and `_FeatureColumn` for derived features.
  for column in schema.derived_columns:
    name = column.name
    retain_lu[name] = column.retain
    which_column = column.WhichOneof("column")
    assert which_column is not None
    if which_column == "embedding":
      config_lu[name] = column.embedding
      column_lu[name] = _embedding_handler(column, column_lu, config_lu)
  elif which_column == "bucketized":
    config_lu[name] = column.bucketized
    column_lu[name] = _bucketized_handler(column, column_lu, config_lu)
  elif which_column == "one_hot":
    config_lu[name] = column.one_hot
    column_lu[name] = _one_hot_handler(column, column_lu, config_lu)

  result_columns = set()
  for name, tf_feature_column in column_lu.items():
    if not retain_lu[name]:
      continue
    result_columns.add(tf_feature_column)
  return set(result_columns)


def _int_fixed_column_handler(column):
  return (tf.contrib.layers.real_valued_column(column_name=column.name,
    dimension=column.int_fixed.dimension, dtype=tf.int64))


  def _int_var_column_handler(column):
    raise NotImplementedError("Unimplemented handler for column: %s" %
        (column.name))


    def _float_fixed_column_handler(column):
      return (tf.contrib.layers.real_valued_column(column_name=column.name,
        dimension=column.float_fixed.dimension, dtype=tf.float32))


      def _float_var_column_handler(column):
        raise NotImplementedError("Unimplemented handler for column: %s" %
            (column.name))


        def _integerized_sparse_column_handler(column):
          integerized_sparse = column.integerized_sparse
  combiner_name = combiner_type_to_string(integerized_sparse.combiner)
  bucket_size = integerized_sparse.bucket_size
  data_type = data_type_to_string(integerized_sparse.data_type)
  return tf.contrib.layers.sparse_column_with_integerized_feature(
      column_name=column.name,
      bucket_size=bucket_size,
      combiner=combiner_name,
      dtype=data_type)


  def _hash_bucket_sparse_column_handler(column):
    sparse_column = column.hash_bucket_sparse
  return tf.contrib.layers.sparse_column_with_hash_bucket(column_name=column.name, hash_bucket_size=sparse_column.hash_bucket_size, combiner=combiner_type_to_string(sparse_column.combiner), dtype=data_type_to_string(sparse_column.data_type))


_NamedConfigColumn = collections.namedtuple(
    "_NamedConfigColumn", ["column_name", "column", "config"])


def _get_depending_columns_or_raise_error(column, column_lu, config_lu):
  """Returns the depending columns for derived column.

  All depending columns must be present in the column lookup table, otherwise errors are raised.

  Args:
    column: (`DerivedFeatureColumn) derived feature column.
    column_lu: (`dict`) maps column name to `_FeatureColumn` instances.
    config_lu: (`dict`) maps column name to `BasicFeatureColumn` or `DerivedFeatureColumn`.

  Raises:
    ValueError: if depending columns are empty.
    KeyError: if any of the depending column is not found in the lookup.

  Returns:
    a `set` of `_NamedConfigColumn` instances.
  """
  if len(column.depending_columns) == 0:
    raise ValueError("Depending columns can't be empty.")
  result_columns = list()
  for name in column.depending_columns:
    if name not in column_lu:
      raise KeyError("Depending column %s not found in column lookup" % (name))
    if name not in config_lu:
      raise KeyError("Depending column %s not found in config lookup" % (name))
    tmp_column = column_lu[name]
    tmp_config = config_lu[name]
    result_columns.append(_NamedConfigColumn(
      column_name=name, column=tmp_column, config=tmp_config))
    return result_columns


def _type_from_list(obj, type_list):
  return any([isinstance(obj, t) for t in type_list])


def _bucketized_column_handler(column, column_lu, config_lu):
  column_config = _get_depending_columns_or_raise_error(
      column, column_lu)[0]
  config = column_config.config
  assert _type_from_list(
      config, [fcs_pb2.IntFixedLenColumn, fcs_pb2.FloatFixedLenColumn])
  # boundaries come from the histogram of the source column.
  boundaries = [b.upper_boundary for b in config.stats.histogram.buckets]
  return tf.contrib.layers.bucketized_column(source_column=column_config.column, boundaries=boundaries)


def _one_hot_column_handler(column, column_lu, conig_lu):
  column_config = _get_depending_columns_or_raise_error(
      column, column_lu, config_lu)[0]
  assert _type_from_list(column_config.config, [
    fcs_pb2.HashBucketSparseColumn, fcs_pb2.IntegerizedSparseColumn])
  return tf.contrib.layers.one_hot_column(sparse_id_column=column_config.column)


def _embedding_handler(column, column_lu, config_lu):
  # Gets depending column config.
  column_config = _get_depending_columns_or_raise_error(
      column, column_lu, config_lu)[0]
  assert _type_from_list(column_config.config, [
    fcs_pb2.HashBucketSparseColumn, fcs_pb2.IntegerizedSparseColumn])
  # Gets embedding column configuration.
  embedding = column.embedding
  return tf.contrib.layers.embedding_column(sparse_id_column=column_config.column, dimension=embedding.dimension, combiner=combiner_type_to_string(embedding.combiner))
