from recsys.util.common import app_init

import argparse
import recsys.util.feature_helper as helper
import logging
import pandas as pd
import re
import tensorflow as tf

def load_item_profile(file):
	item_df = pd.read_csv(filepath_or_buffer = file)
	# transform the data frame into a dictionary where the key is the movie id
	item_dict = {}
	title_re = re.compile('(^.*?)\s+\((\d+)\)$')
	for _, row in item_df.iterrows():
		matches = title_re.match(row['title'])
		title = matches.group(1)
		year = matches.group(2)
		genres = row['genres'].split('|')
		item_dict[row['movieId']] = {"title": title.lower(), "year": year, "genres" : [e.lower() for e in genres]}
	logging.debug(item_dict)
	return item_dict

def convert_to_tf_records(rating_file, tf_record_file):
	ratings_df = pd.read_csv(filepath_or_buffer = rating_file)
	with tf.python_io.TFRecordWriter(tf_record_file) as writer:
		for _, row in ratings_df.iterrows():
			user_id = str(int(row["userId"]))
			item_id = str(int(row["movieId"]))
			rating = row["rating"]
			example = helper.example_from_feature_dict({
					"user_id" : helper.string_feature(user_id),
					"item_id" : helper.string_feature(item_id),
					"rating" : helper.float_feature(rating)
				})
			writer.write(example.SerializeToString())

