from functools import reduce
from types import GeneratorType
import array
import collections
import logging
import os

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin
from tqdm import *
import numpy as np

from wort.core.config_registry import ConfigRegistry
from wort.core.io_handler import IOHandler


class NGramLanguageModel(BaseEstimator, VectorizerMixin):
	def __init__(self, min_frequency=0, lowercase=True, stop_words=None, encoding='utf-8', max_features=None,
				 preprocessor=None, tokenizer=None, analyzer='word', token_pattern=r'(?u)\b\w\w+\b', decode_error='strict',
				 strip_accents=None, input='content', ngram_range=(1, 3), random_state=1105, cds=1.0,
				 cache_intermediary_results=True, subsampling_rate=None, cache_path='~/.wort_data/model_cache',
				 log_level=logging.INFO, log_file=None):
		self.min_frequency = min_frequency
		self.lowercase = lowercase
		self.stop_words = stop_words
		self.encoding = encoding
		self.max_features = max_features
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.analyzer = analyzer
		self.token_pattern = token_pattern
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.input = input
		self.ngram_range = ngram_range
		self.random_state = random_state
		self.cds = cds
		self.subsampling_rate = subsampling_rate
		self.cache_intermediary_results = cache_intermediary_results
		if (cache_path is not None and cache_path.startswith('~')):
			cache_path = os.path.expanduser(cache_path)
		if (not os.path.exists(cache_path)):
			os.makedirs(cache_path)
		self.cache_path = cache_path
		self.nn = None

		self.inverted_index_ = collections.defaultdict(dict)
		self.index_ = collections.defaultdict(dict)
		self.c_w_ = None
		self.vocab_count_ = 0
		self.token_count_ = 0
		self.M_ = None
		self.T_ = None
		self.density_ = 0.

		'''
		self.config_registry_ = ConfigRegistry(path=cache_path, min_frequency=self.min_frequency, lowercase=self.lowercase,
											   stop_words=self.stop_words, encoding=self.encoding, max_features=self.max_features,
											   preprocessor=self.preprocessor, tokenizer=self.tokenizer, analyzer=self.analyzer,
											   token_pattern=self.token_pattern, decode_error=self.decode_error,
											   strip_accents=self.strip_accents, input=self.input, ngram_range=self.ngram_range,
											   random_state=self.random_state, subsampling_rate=self.subsampling_rate)

		self.io_handler_ = IOHandler(cache_path=cache_path, log_file=log_file, log_level=log_level)
		self.io_handler_.setup_logging()
		'''

	def _delete_from_vocab(self, W, idx):
		W = np.delete(W, idx)

		for i in idx:
			item = self.index_[i]
			del self.inverted_index_[item]
			del self.index_[i]

		return W

	def _n_gram_size(self, feat):
		return feat.count(' ') + 1 # FIXME - THIS IS BOUND TO FAIL BECAUSE ITS RELYING ON A WHITESPACE AS A BOUNDARY BETWEEN TWO WORDS!!!!!!!!!!!!!!!!!!!

	def fit_vocabulary(self, raw_documents, analyser=None):
		if (analyser is None):
			analyser = self.build_analyzer()

		n_vocab = collections.defaultdict(int)
		w = {}

		# Extract vocabulary
		for doc in tqdm(raw_documents):
			for feature in analyser(doc):
				print('FEAT: {}; N GRAM SIZE: {}'.format(feature, self._n_gram_size(feature)))
				ngram_size = self._n_gram_size(feature)

				idx = self.inverted_index_[ngram_size].get(feature, n_vocab[ngram_size] + 1)

				# Build vocab
				if (idx > n_vocab[ngram_size]):
					n_vocab[ngram_size] += 1
					self.inverted_index_[ngram_size][feature] = n_vocab[ngram_size]
					wi = w.get(ngram_size, array.array('i'))
					wi.append(1)
					w[ngram_size] = wi
				else:
					w[ngram_size][idx] += 1

		import json
		print(json.dumps(self.inverted_index_, indent=4))

		# Vocab was used for indexing (hence, started at 0 for the first item (NOT init!)), so has to be incremented by 1
		# to reflect the true vocab count
		for r in range(self.ngram_range[0], self.ngram_range[1]+1):
			n_vocab[r] += 1
			self.index_[r] = dict(zip(self.inverted_index_[r].values(), self.inverted_index_[r].keys()))
			w[r] = np.array(w[r], dtype=np.uint64)
		logging.info('Finished Extracting vocabulary! n_vocab={}'.format(n_vocab))

		logging.info('Filtering extremes...')
		token_count = {}
		for r in range(self.ngram_range[0], self.ngram_range[1]+1):
			# Filter extremes
			if (self.min_frequency > 1):
				idx = np.where(w[r] < self.min_frequency)[0]

				w[r] = self._delete_from_vocab(w[r], idx)

				n_vocab[r] -= len(idx)

			# Max Features Filter
			if (self.max_features is not None and self.max_features < n_vocab):
				idx = np.argpartition(-w[r], self.max_features)[self.max_features:]

				w[r] = self._delete_from_vocab(w[r], idx)

				n_vocab[r] -= len(idx)

			token_count[r] = w[r].sum()
			# Subsampling TODO: this can certainly be optimised
			'''
			if (self.subsampling_rate is not None):
				rnd = np.random.RandomState(self.random_state)
				t = self.subsampling_rate * token_count

				cand_idx = np.where(W > t)[1]  # idx of words exceeding threshold

				P = 1 - np.sqrt(W * (1 / t))  # `word2vec` subsampling formula
				R = rnd.rand(W.shape)

				subsample_idx = np.where(R <= P)[1]  # idx of filtered words

				idx = cand_idx - subsample_idx

				W = self._delete_from_vocab(W, idx)

				n_vocab -= len(idx)
				'''
		logging.info('Finished Filtering extremes! n_vocab={}; n_tokens={}'.format(n_vocab, token_count))

		self.c_w_ = w
		self.vocab_count_ = n_vocab
		self.token_count_ = token_count

		# Watch out when rebuilding the index, `self.index_` needs to be built _before_ `self.inverted_index_`
		# to reflect the updated `W` array
		logging.info('Rebuilding indices...')
		for r in range(self.ngram_range[0], self.ngram_range[1] + 1):
			self.index_[r] = dict(zip(range(n_vocab[r]), self.index_[r].values()))
			self.inverted_index_[r] = dict(zip(self.index_[r].values(), self.index_[r].keys()))
		logging.info('Indices rebuilt!')

	def fit_ngram_probabilities(self): # TODO: Why not calculate at prediction time? Probably enough to have the unigram probabilities calculated and the other ngram extracted and counted, and then can go from there
		p_w_c = collections.defaultdict(lambda: collections.defaultdict(float))

		# Unigram probabilities
		for w in self.inverted_index_[1].keys():
			p_w_c[1][w] = int(self.c_w_[1][self.inverted_index_[1][w]]) / self.token_count_[1]

		# > 2-gram probabilities
		for r in range(*self.ngram_range):
			for wa in self.inverted_index_[r].keys():
				c = reduce(lambda acc, ngram: acc + 1 if ngram.rsplit(' ', 1)[0] == wa else acc, self.inverted_index_[r + 1].keys(), 0) # Count C(wb, wa) => C(wb | wa), extends to any ngram size
				# TODO: THE INDEXING IS WRONG, IT NEEDS TO BE INDEXED WITH WB AS WELL!!!!!
				p_w_c[r + 1][wa] = c / int(self.c_w_[r][self.inverted_index_[r][wa]]) # P(wb | wa) = C(wb, wa) / C(wa), extends to any ngram size

		import json
		print(json.dumps(p_w_c, indent=4))



	def fit(self, raw_documents, y=None):

		# Shameless copy/paste from Radims word2vec Tutorial, no generators matey, need multi-pass!!!
		if (raw_documents is not None):
			if (isinstance(raw_documents, GeneratorType)):
				raise TypeError('You can\'t pass a generator as the sentences argument. Try an iterator.')

		analyser = self.build_analyzer()

		##### FIT VOCABULARY
		vocab_folder = None #self.config_registry_.vocab_cache_folder()
		if (vocab_folder is not None and vocab_folder != ''):
			# Load cached resources
			logging.info('Loading cached vocabulary resources from {}...'.format(os.path.join(self.cache_path, vocab_folder)))
			self.p_w_ = self.io_handler_.load_p_w(vocab_folder)
			self.vocab_count_ = self.io_handler_.load_vocab_count(vocab_folder)
			self.token_count_ = self.io_handler_.load_token_count(vocab_folder)
			self.index_ = self.io_handler_.load_index(vocab_folder)
			self.inverted_index_ = self.io_handler_.load_inverted_index(vocab_folder)
			logging.info('Cache loaded!')
		else:
			# Create vocabulary
			logging.info('Fitting vocabulary...')
			self.fit_vocabulary(raw_documents=raw_documents, analyser=analyser)
			logging.info('Vocabulary fitted!')

			# Cache vocabulary
			if (self.cache_intermediary_results):
				sub_folder = self.config_registry_.register_vocab()
				logging.info('Storing vocabulary cache to folder {}...'.format(sub_folder))
				self.io_handler_.save_index(self.index_, sub_folder)
				self.io_handler_.save_inverted_index(self.inverted_index_, sub_folder)
				self.io_handler_.save_vocab_count(self.vocab_count_, sub_folder)
				self.io_handler_.save_token_count(self.token_count_, sub_folder)
				self.io_handler_.save_p_w(self.p_w_, sub_folder)
				logging.info('Cache stored!')
		##################

		##### FIT N-GRAM PROBABILITIES
		cooc_folder = None #self.config_registry_.cooccurrence_matrix_folder()
		if (cooc_folder is not None and cooc_folder != ''):
			logging.info('Loading cached co-occurrence matrix resources from {}...'.format(os.path.join(self.cache_path, cooc_folder)))
			self.M_ = self.io_handler_.load_cooccurrence_matrix(cooc_folder)
			logging.info('Cache loaded!')
		else:
			logging.info('Fitting ngram probabilities...')
			self.fit_ngram_probabilities()
			logging.info('Ngram probabilites fitted!')

			# Cache co-occurrence matrix
			if (self.cache_intermediary_results):
				sub_folder = self.config_registry_.register_cooccurrence_matrix()
				logging.info('Storing co-occurrence matrix cache to folder {}...'.format(sub_folder))
				self.io_handler_.save_cooccurrence_matrix(self.M_, sub_folder)
				logging.info('Cache stored!')
		##################

		##### FIT PMI FEATURE TRANSFORMATION
		pmi_folder = self.config_registry_.pmi_matrix_folder()
		if (pmi_folder is not None and pmi_folder != ''):
			logging.info('Loading cached PMI matrix resources from {}...'.format(os.path.join(self.cache_path, pmi_folder)))
			self.T_ = self.io_handler_.load_pmi_matrix(pmi_folder)
			logging.info('Cache loaded!')
		else:
			logging.info('Fitting PMI matrix...')
			self.fit_pmi_matrix()
			logging.info('PMI matrix fitted!')

			# Cache PMI matrix
			if (self.cache_intermediary_results):
				sub_folder = self.config_registry_.register_pmi_matrix()
				logging.info('Storing PMI matrix cache to folder {}...'.format(sub_folder))
				self.io_handler_.save_pmi_matrix(self.T_, sub_folder)
				logging.info('Cache stored!')
		##################

		##### FIT DIMENSIONALITY REDUCTION
		if (self.context_vector_integration is not None):
			if (isinstance(self.context_vector_integration, Callable)):
				context_vector_integration_fn = self.context_vector_integration
			else:
				context_vector_integration_fn = getattr(context_vector_integration, '{}_context_vectors'.format(self.context_vector_integration))
		else:
			context_vector_integration_fn = None
		##################

		##### ADD CONTEXT VECTORS (OR NOT)
		if (self.dim_reduction is not None):
			logging.info('Fitting dimensionality reduction...')
			W, C = self.fit_dimensionality_reduction()
			logging.info('Dimensionality reduction fitted!')

			# Add context vectors
			if (context_vector_integration_fn is not None):
				self.T_ = context_vector_integration_fn(W=W, C=C, **self.context_vector_integration_kwargs)
			else:
				self.T_ = W
		else:
			# Add context vectors for the sparse case
			if (context_vector_integration_fn is not None):
				self.T_ = context_vector_integration_fn(W=self.T_.tolil(), C=self.T_.transpose().tolil()).tocsr()
		##################

		return self