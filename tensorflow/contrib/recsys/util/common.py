import argparse
import logging
import sys

def app_init():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--log_level',
		type=str,
		default='INFO',
		help='logging level.'
	)
	parsed,unparsed = parser.parse_known_args()
	numeric_level = getattr(logging, parsed.log_level.upper(), None)
	if not isinstance(numeric_level, int):
		raise ValueError('Invalid log level: %s' % log_level)
	logging.basicConfig(level = numeric_level)
	logging.info("app_init done")
	return (sys.argv[0], unparsed)
