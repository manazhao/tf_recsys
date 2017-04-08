import tensorflow as tf

def int64_feature(val):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[val]))

def int64_list_feature(val_list):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=val_list))

def bytes_feature(val):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value=[val]))

def bytes_list_feature(val_list):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value=val_list))

def float_feature(val):
	return tf.train.Feature(float_list = tf.train.FloatList(value=[val]))

def float_list_feature(val_list):
	return tf.train.Feature(float_list = tf.train.FloatList(value=val_list))

def string_feature(str):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value=[str.encode('utf-8')]))
def string_list_feature(str_list):
	str_bytes_list = [k.encode('utf-8') for k in str_list]
	return bytes_list_feature(str_bytes_list)

# Constructs a tf.Example with feature dictionary where key is feature name and
# value is tf.train.Feature
def example_from_feature_dict(feature_dict):
	return tf.train.Example(features = tf.train.Features(feature = feature_dict))
