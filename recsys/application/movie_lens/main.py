import argparse
from recsys.application.movie_lens.movie_lens_data import *
from recsys.util.common import *

if __name__ == '__main__':
	prog_name, unparsed = app_init()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--operation',
		type = str,
		default = ''
	)
	parser.add_argument(
		'--rating_file',
		type = str,
		default = ''
	)
	parser.add_argument(
		'--example_file',
		type = str,
		default = ''
	)
	FLAGS, _ = parser.parse_known_args(unparsed)
	if FLAGS.operation == 'generate_example':
		logging.info("convert rating file {} to tf.Examples {}".format(FLAGS.rating_file, FLAGS.example_file))
		convert_to_tf_records(FLAGS.rating_file, FLAGS.example_file)

