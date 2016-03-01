__author__ = 'thomas'
import csv
import gzip
import sys

import joblib

csv.field_size_limit(sys.maxsize)


class GzipStreamReader(object):
	def __init__(self, path, lowercase=True):
		self.path_ = path
		self.lowercase_ = lowercase

	def __iter__(self):
		with gzip.open(self.path_, 'rt', encoding='utf-8', errors='replace') as in_file:
			for line in in_file:
				processed_line = line if not self.lowercase_ else line.lower()

				yield processed_line


class CSVStreamReader(object):
	def __init__(self, path, delimiter=','):
		self.path_ = path
		self.delimiter = delimiter

	def __iter__(self):
		with open(self.path_, 'r') as wiki_file:
			csv_reader = csv.reader(wiki_file, delimiter=self.delimiter)

			for line in csv_reader:
				yield line[0]


class MovieReviewReader(object):
	def __init__(self, path):
		self.path_ = path

	def __iter__(self):
		unlabelled_docs = joblib.load(self.path_)

		for doc in unlabelled_docs:
			yield doc.replace('<br />', ' ')


class FrostReader(object):
	def __init__(self, path):
		self.path_ = path

	def __iter__(self):
		with open(self.path_, 'r') as frost:
			for line in frost:
				yield line