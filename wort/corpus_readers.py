__author__ = 'thomas'
import csv
import sys

import joblib

csv.field_size_limit(sys.maxsize)


class WikipediaReader(object):
	def __init__(self, path):
		self.path_ = path

	def __iter__(self):
		with open(self.path_, 'r') as wiki_file:
			csv_reader = csv.reader(wiki_file)

			for line in csv_reader:
				yield line[1]


class MovieReviewReader(object):
	def __init__(self, path):
		self.path_ = path

	def __iter__(self):
		unlabelled_docs = joblib.load(self.path_)

		for doc in unlabelled_docs:
			yield doc.replace('<br />', ' ')