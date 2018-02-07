__author__ = 'thomas'
import csv
import gzip
import sys

csv.field_size_limit(sys.maxsize)


class TextStreamReader(object):
	def __init__(self, path, lowercase=True, encoding='utf-8'):
		self.path_ = path
		self.lowercase_ = lowercase
		self.encoding_ = encoding

	def __iter__(self):
		with open(self.path_, 'r', encoding=self.encoding_) as text_file:
			for line in text_file:
				processed_line = line.strip() if not self.lowercase_ else line.strip().lower()

				yield processed_line


class CoNLLStreamReader(object):
	def __init__(self, path, data_index, order='seq', lowercase=True, sep='\t', num_columns=7, head_idx=-2, token_idx=0,
				 encoding='utf-8'): # order supports 'dep' (dependency order) or 'seq' (standard sequential order)
		self.path_ = path
		self.data_index_ = data_index
		self.order_ = order
		self.lowercase_ = lowercase
		self.sep_ = sep
		self.num_columns_ = num_columns
		self.head_idx_ = head_idx
		self.token_idx_ = token_idx
		self.encoding_ = encoding

	def __iter__(self):
		with open(self.path_, 'r', encoding=self.encoding_) as conll_file:
			curr_line = []
			root_idx = -1
			for line in conll_file:
				parts = line.strip().split(self.sep_)
				if (len(parts) < self.num_columns_):
					if (self.order_ == 'seq'):
						sent = ' '.join(map(lambda x: x[self.data_index_], curr_line))
						sent = sent.lower() if self.lowercase_ else sent
						curr_line = []
					else: # Dep Context, run a BFS over the dependency tree
						q = [root_idx]
						seq = []
						already_seen = set()

						while (len(q) > 0):
							item = q.pop(0)

							if (item not in already_seen):
								seq.append(item)
								already_seen.add(item)

								for idx, p in enumerate(curr_line):
									if (p[self.head_idx_] != '_' and int(p[self.head_idx_])-1 == item):
										q.append(idx)

						sent = ' '.join(map(lambda x: curr_line[x][self.data_index_], seq))
						sent = sent.lower() if self.lowercase_ else sent
						curr_line = []

					yield sent
				else:
					curr_line.append(parts)
					if (parts[self.head_idx_] == '0'):
						root_idx = int(parts[self.token_idx_]) - 1


class GzipStreamReader(object):
	def __init__(self, path, lowercase=True, encoding='utf-8'):
		self.path_ = path
		self.lowercase_ = lowercase
		self.encoding_ = encoding

	def __iter__(self):
		with gzip.open(self.path_, 'rt', encoding=self.encoding_, errors='replace') as in_file:
			for line in in_file:
				processed_line = line if not self.lowercase_ else line.lower()

				yield processed_line


class CSVStreamReader(object):
	def __init__(self, path, lowercase=True, delimiter=',', data_index=0, encoding='utf-8'):
		self.path_ = path
		self.delimiter = delimiter
		self.data_index = data_index
		self.lowercase_ = lowercase
		self.encoding_ = encoding

	def __iter__(self):
		with open(self.path_, 'r', encoding=self.encoding_) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=self.delimiter)

			for line in csv_reader:
				processed_line = line[self.data_index] if not self.lowercase_ else line[self.data_index].lower()

				yield processed_line