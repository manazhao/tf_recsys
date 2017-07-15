import random
import tensorflow as tf

def number_of_tf_records(tf_record_file):
	i = 0
	for _ in tf.python_io.tf_record_iterator(tf_record_file):
		i += 1
	return i

def generate_cv_partitions(train_file, cv_fold = 5):
	cv_files = ['{}-cv-{:03d}-{:03d}'.format(train_file, i, cv_fold) for i in range(cv_fold)]
	writers = [tf.python_io.TFRecordWriter(cv_files[i]) for i in range(cv_fold)]
	for s_example in tf.python_io.tf_record_iterator(train_file):
		writers[random.randint(0, cv_fold - 1)].write(s_example)
	for writer in writers:
		writer.close()
	return cv_files

def split_train_test(data_file, train_ratio):
	train_file = '{}-train-{:.2f}'.format(data_file,round(train_ratio,2))
	test_file = '{}-test-{:.2f}'.format(data_file,1 - round(train_ratio,2))
	train_writer = tf.python_io.TFRecordWriter(train_file)
	test_writer = tf.python_io.TFRecordWriter(test_file)
	for s_example in tf.python_io.tf_record_iterator(data_file):
		if random.random() < train_ratio:
			train_writer.write(s_example)
		else:
			test_writer.write(s_example)
	train_writer.close()
	test_writer.close()
	return (train_file, test_file)
