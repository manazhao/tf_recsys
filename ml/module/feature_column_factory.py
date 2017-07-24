from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
    a set of feature columns.
  """
  schema = feature_column_schema_from_config(schema_pbtxt)
  name_to_config_lookup = dict()
  name_to_column_lookup = dict()

  for column in schema.basic_features:
    name = column.name
    name_to_config_lookup[name] = column
    # Skips label feautre column if include_target is false.
    if name == FEATURE_LABEL_NAME and not include_target:
      continue
    which_column = column.WhichOneof("column")
    assert which_column is not None
    # Gets `_FeatureColumn` for the oneof fields of `BasicFeatureColumn`.
    if which_column == "int_fixed":
      name_to_column_lookup[name] = _int_fixed_column_handler(column)
    elif which_column == "int_var":
      name_to_column_lookup[name] = _int_var_column_handler(column)
    elif which_column == "float_fixed":
      name_to_column_lookup[name] = _float_fixed_column_handler(column)
    elif which_column == "float_var":
      name_to_column_lookup[name] = _float_var_column_handler(column)
    elif which_column == "integerized_sparse":
      name_to_column_lookup[name] = _integerized_sparse_column_handler(column)
    elif which_column == "hash_bucket_sparse":
      name_to_column_lookup[name] = _hash_bucket_sparse_column_handler(column)
    else:
      raise ValueError("unimplemented handler for column: %s" % (which_column))
  result_columns = list()
  for name, tf_feature_column in name_to_column_lookup.items():
    if not name_to_config_lookup[name].retain:
      continue
    result_columns.append(tf_feature_column)
  return set(result_columns)


def _int_fixed_column_handler(column):
  return tf.contrib.layers.real_valued_column(column_name=column.name, dimension=column.int_fixed.dimension, dtype=tf.int64)


def _int_var_column_handler(column):
  raise NotImplementedError("Unimplemented handler for column: %s" %
                            (column.name))


def _float_fixed_column_handler(column):
  return tf.contrib.layers.real_valued_column(column_name=column.name, dimension=column.float_fixed.dimension, dtype=tf.float32)


def _float_var_column_handler(column):
  raise NotImplementedError("Unimplemented handler for column: %s" %
                            (column.name))


def _integerized_sparse_column_handler(column):
  integerized_sparse = column.integerized_sparse
  combiner_name = combiner_type_to_string(integerized_sparse.combiner)
  bucket_size = integerized_sparse.bucket_size
  data_type = data_type_to_string(integerized_sparse.data_type)
  return tf.contrib.layers.sparse_column_with_integerized_feature(column_name=column.name, bucket_size=bucket_size, combiner=combiner_name, dtype=data_type)


def _hash_bucket_sparse_column_handler(column):
  sparse_column = column.hash_bucket_sparse
  return tf.contrib.layers.sparse_column_with_hash_bucket(column_name=column.name, hash_bucket_size=sparse_column.hash_bucket_size, combiner=combiner_type_to_string(sparse_column.combiner), dtype=data_type_to_string(sparse_column.data_type))
