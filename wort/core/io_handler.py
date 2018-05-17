__author__ = 'thomas'
import json
import logging
import os
import sys

import joblib

from wort.core import utils


class IOHandler(object):
	def __init__(self, cache_path, log_file, log_level, log_format='%(asctime)s: %(levelname)s - %(message)s',
				 date_format='[%d/%m/%Y %H:%M:%S %p]', base_config_file='~/.wort_data/config/config.json'):
		self.cache_path_ = cache_path
		self.log_file_ = log_file
		self.log_level_ = log_level
		self.log_format_ = log_format
		self.date_format_ = date_format

		if (base_config_file.startswith('~')):
			base_config_file = os.path.expanduser(base_config_file)
		if (not os.path.exists(os.path.split(base_config_file)[0])):
			os.makedirs(os.path.split(base_config_file)[0])
			config = {
				'mem_proportion': 0.8,
				'num_chunks': 10,
				'dtype_size': 64
			}
			with open(base_config_file, 'w') as config_file:
				json.dump(config, config_file)
		else:
			with open(base_config_file, 'r') as config_file:
				config = json.load(config_file)

		self.base_config_ = config

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

	def save_p_w(self, p_w, sub_folder, base_path=None):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		if (base_path is None):
			base_path = self.cache_path_

		utils.numpy_to_hdf(p_w, os.path.join(base_path, sub_folder), 'p_w')

	def load_vocab_count(self, sub_folder):
		return joblib.load(os.path.join(self.cache_path_, sub_folder, 'vocab_count.joblib'))['vocab_count']

	def save_vocab_count(self, vocab_count, sub_folder):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		joblib.dump({'vocab_count': vocab_count}, os.path.join(self.cache_path_, sub_folder, 'vocab_count.joblib'), compress=3)

	def load_token_count(self, sub_folder):
		return joblib.load(os.path.join(self.cache_path_, sub_folder, 'token_count.joblib'))['token_count']

	def save_token_count(self, token_count, sub_folder):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		joblib.dump({'token_count': token_count}, os.path.join(self.cache_path_, sub_folder, 'token_count.joblib'), compress=3)

	def load_index(self, sub_folder):
		return joblib.load(os.path.join(self.cache_path_, sub_folder, 'index.joblib'))

	def save_index(self, index, sub_folder, base_path=None):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		if (base_path is None):
			base_path = self.cache_path_

		joblib.dump(index, os.path.join(base_path, sub_folder, 'index.joblib'), compress=3)

	def load_inverted_index(self, sub_folder):
		return joblib.load(os.path.join(self.cache_path_, sub_folder, 'inverted_index.joblib'))

	def save_inverted_index(self, inverted_index, sub_folder, base_path=None):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		if (base_path is None):
			base_path = self.cache_path_

		joblib.dump(inverted_index, os.path.join(base_path, sub_folder, 'inverted_index.joblib'), compress=3)

	def load_similarity_matrix(self, sub_folder):
		p = os.path.join(self.cache_path_, sub_folder, 'S.hdf')
		if (not os.path.exists(p)):
			logging.warning('No similarity matrix found at path={}!'.format(p))
			return None
		return utils.hdf_to_sparse_csx_matrix(os.path.join(self.cache_path_, sub_folder), 'S.hdf', sparse_format='csr')

	def save_similarity_matrix(self, S, sub_folder, base_path=None):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		if (base_path is None):
			base_path = self.cache_path_

		utils.sparse_matrix_to_hdf(S, os.path.join(base_path, sub_folder), 'S.hdf')

	def load_context_selection_matrix(self, sub_folder):
		p = os.path.join(self.cache_path_, sub_folder, 'C.hdf')
		if (not os.path.exists(p)):
			logging.warning('No context selection matrix found at path={}!'.format(p))
			return None
		return utils.hdf_to_sparse_csx_matrix(os.path.join(self.cache_path_, sub_folder), 'C.hdf', sparse_format='csr')

	def save_context_selection_matrix(self, C, sub_folder, base_path=None):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		if (base_path is None):
			base_path = self.cache_path_

		utils.sparse_matrix_to_hdf(C, os.path.join(base_path, sub_folder), 'C.hdf')

	def load_cooccurrence_matrix(self, sub_folder):
		p = os.path.join(self.cache_path_, sub_folder, 'M.hdf')
		if (not os.path.exists(p)):
			logging.warning('No co-occurrence matrix found at path={}!'.format(p))
			return None
		return utils.hdf_to_sparse_csx_matrix(os.path.join(self.cache_path_, sub_folder), 'M.hdf', sparse_format='csr')

	def save_cooccurrence_matrix(self, M, sub_folder, base_path=None):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		if (base_path is None):
			base_path = self.cache_path_

		utils.sparse_matrix_to_hdf(M, os.path.join(base_path, sub_folder), 'M.hdf')

	def load_pmi_matrix(self, sub_folder):
		return utils.hdf_to_sparse_csx_matrix(os.path.join(self.cache_path_, sub_folder), 'T.hdf', sparse_format='csr')

	def save_pmi_matrix(self, T, sub_folder, base_path=None):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		if (base_path is None):
			base_path = self.cache_path_
		utils.sparse_matrix_to_hdf(T, os.path.join(base_path, sub_folder), 'T.hdf')

	def load_context_representation_matrix(self, sub_folder):
		return utils.hdf_to_numpy(os.path.join(self.cache_path_, sub_folder), 'O.hdf')

	def save_context_representation_matrix(self, O, sub_folder, base_path=None):
		if (not os.path.exists(os.path.join(self.cache_path_, sub_folder))):
			os.makedirs(os.path.join(self.cache_path_, sub_folder))

		if (base_path is None):
			base_path = self.cache_path_
		utils.numpy_to_hdf(O, os.path.join(base_path, sub_folder), 'O.hdf')

	def save_model_properties(self, properties, sub_folder, base_path=None):
		if (base_path is None):
			base_path = self.cache_path_

		joblib.dump(properties, os.path.join(base_path, sub_folder, 'model_properties.joblib'), compress=3)

	def load_model_properties(self, sub_folder):
		p = os.path.join(self.cache_path_, sub_folder, 'model_properties.json')
		if (not os.path.exists(p)):
			logging.warning('Model property file not found at path={}!'.format(p))
			return None

		return joblib.load(os.path.join(self.cache_path_, sub_folder, 'model_properties.joblib'))