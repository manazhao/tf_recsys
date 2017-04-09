import logging
import tensorflow as tf
import recsys.util.proto.config_pb2 as config

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

def get_int64_feature(example, feature_name):
	return example.features.feature[feature_name].int64_list.value[0]

def get_int64_list_feature(example, feature_name):
	return list(example.features.feature[feature_name].int64_list.value)

def get_float_feature(example, feature_name):
	return example.features.feature[feature_name].float_list.value[0]

def get_float_list_feature(example, feature_name):
	return list(example.features.feature[feature_name].float_list.value)

def get_bytes_feature(example, feature_name):
	return example.features.feature[feature_name].bytes_list.value[0]

def get_bytes_list_feature(example, feature_name):
	return example.features.feature[feature_name].bytes_list.value

def get_string_feature(example, feature_name):
	return example.features.feature[feature_name].bytes_list.value[0].decode('utf-8')

def get_string_list_feature(example, feature_name):
	return [s.decode('utf-8') for s in example.features.feature[feature_name].bytes_list.value]


# Reads batched features and labels from given files, and consumes them through
# callback function "consum_batch_fn".
# feature_spec: dictionary specifying the type of each feature.
# input_config: settings for generating batched features and labels.
# consume_batch_fn: callback function that defines the consumption of the 
# batched features and labels.
def fetch_and_process_features(filenames, feature_spec, input_config, consume_batch_fn):
	# Reads examples from the filenames and parse them into features.
	def _read_and_decode(filename_queue, feature_spec, batch_size = 2, capacity = 30, num_threads = 2, min_after_dequeue = 10):
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(
			serialized_example, features = feature_spec)
		batched_features = tf.train.shuffle_batch(features,
			batch_size = batch_size,
			capacity = capacity,
			num_threads = num_threads,
			min_after_dequeue = min_after_dequeue)
		return batched_features

	filename_queue = tf.train.string_input_producer(
		filenames, num_epochs = input_config.num_epochs)
	features = _read_and_decode(
		filename_queue,
		feature_spec,
		batch_size = input_config.batch_size,
		capacity = input_config.capacity,
		num_threads = input_config.num_threads,
		min_after_dequeue = input_config.min_after_dequeue
		)
	init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
	with tf.Session()  as sess:
	    sess.run(init_op)
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(coord=coord)
	    for i in range(input_config.num_batches):
	    	logging.info('current batch:{}'.format(i))
	    	consume_batch_fn(sess, features)
	    coord.request_stop()
	    coord.join(threads)