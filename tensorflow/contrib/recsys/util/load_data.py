import feature
import pandas as pd
import re
import logging

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

def convert_to_example(rating_file):
	ratings_df = pd.read_csv(filepath_or_buffer = rating_file)
	examples = []
	for _, row in ratings_df.iterrows():
		user_id = row["userId"]
		item_id = row["movieId"]
		rating = row["rating"]
		example = feature.example_from_feature_dict({
				"user_id" : feature.bytes_feature(user_id),
				"item_id" : feature.bytes_feature(item_id),
				"rating" : feature.float_feature(rating)
			})
		examples.extend(example)
	return examples


