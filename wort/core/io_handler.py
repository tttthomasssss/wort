__author__ = 'thomas'
import logging
import os
import sys

import joblib

from wort.core import utils


class IOHandler(object):
	def __init__(self, cache_path, log_file, log_level, log_format='%(asctime)s: %(levelname)s - %(message)s',
				 date_format='[%d/%m/%Y %H:%M:%S %p]'):
		self.cache_path_ = cache_path
		self.log_file_ = log_file
		self.log_level_ = log_level
		self.log_format_ = log_format
		self.date_format_ = date_format

	def setup_logging(self):
		log_formatter = logging.Formatter(fmt=self.log_format_, datefmt=self.date_format_)
		root_logger = logging.getLogger()
		root_logger.setLevel(self.log_level_)

		# stdout logging
		if (len(root_logger.handlers) <= 0):
			console_handler = logging.StreamHandler(sys.stdout)
			console_handler.setFormatter(log_formatter)
			root_logger.addHandler(console_handler)

			# file logging
			if (self.log_file_ is not None):
				log_path = os.path.split(self.log_file_)[0]
				if (not os.path.exists(log_path)):
					os.makedirs(log_path)

				file_handler = logging.FileHandler(self.log_file_)
				file_handler.setFormatter(log_formatter)
				root_logger.addHandler(file_handler)

	def load_p_w(self, sub_folder):
		if (os.path.exists(os.path.join(os.path.join(self.cache_path_, sub_folder, 'p_w.hdf')))):
			p_w = utils.hdf_to_numpy(os.path.join(self.cache_path_, sub_folder), 'p_w')
		else:
			p_w = joblib.load(os.path.join(self.cache_path_, sub_folder, 'p_w.joblib'))

		return p_w

	def save_p_w(self, p_w, sub_folder):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		utils.numpy_to_hdf(p_w, os.path.join(self.cache_path_, sub_folder), 'p_w')

	def load_vocab_count(self, sub_folder):
		return joblib.load(os.path.join(self.cache_path_, sub_folder, 'vocab_count.joblib'))['vocab_count']

	def save_vocab_count(self, vocab_count, sub_folder):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		joblib.dump({'vocab_count': vocab_count}, os.path.join(self.cache_path_, sub_folder, 'vocab_count.joblib'), compress=3)

	def load_index(self, sub_folder):
		return joblib.load(os.path.join(self.cache_path_, sub_folder, 'index.joblib'))

	def save_index(self, index, sub_folder):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		joblib.dump(index, os.path.join(self.cache_path_, sub_folder, 'index.joblib'), compress=3)

	def load_inverted_index(self, sub_folder):
		return joblib.load(os.path.join(self.cache_path_, sub_folder, 'inverted_index.joblib'))

	def save_inverted_index(self, inverted_index, sub_folder):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		joblib.dump(inverted_index, os.path.join(self.cache_path_, sub_folder, 'inverted_index.joblib'), compress=3)

	def load_cooccurrence_matrix(self, sub_folder):
		return utils.hdf_to_sparse_csx_matrix(os.path.join(self.cache_path_, sub_folder), 'M.hdf', sparse_format='csr')