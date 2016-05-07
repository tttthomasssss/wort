__author__ = 'thomas'
import csv
import gzip
import sys

csv.field_size_limit(sys.maxsize)


class TextStreamReader(object):
	def __init__(self, path, lowercase=True):
		self.path_ = path
		self.lowercase_ = lowercase

	def __iter__(self):
		with open(self.path_, 'r') as text_file:
			for line in text_file:
				processed_line = line.strip() if not self.lowercase_ else line.strip().lower()

				yield processed_line


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
		with open(self.path_, 'r') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=self.delimiter)

			for line in csv_reader:
				yield line[0]