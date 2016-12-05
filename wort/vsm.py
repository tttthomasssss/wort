__author__ = 'thomas'
from collections import Callable
from types import GeneratorType
import array
import logging
import math
import os

from scipy import sparse
from scipy.sparse import sputils
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.neighbors import NearestNeighbors
from tqdm import *
import numpy as np

from wort.core import context_vector_integration
from wort.core import context_weighting
from wort.core import dim_reduction
from wort.core import feature_transformation
from wort.core import oov_handler
from wort.core import vector_composition
from wort.core.config_registry import ConfigRegistry
from wort.core.io_handler import IOHandler
from wort.core.utils import determine_chunk_size

# TODO: SVD based on http://www.aclweb.org/anthology/Q/Q15/Q15-1016.pdf, esp. chapter 7, practical recommendations
	# Subsampling
	# Normalisation
	# Hellinger PCA (do we need that?)
	# NMF as an alternative to SVD (sklearns NMF is _very_ slow for dimensions > 100!!)
	# Support min_df and max_df
	# Optimise the shizzle-whizzle
	# Memmap option?
	# Improve numerical precision
	# Support other thresholds
	# Better sklearn pipeline support (e.g. get_params())
	# Compress hdf output
	# Cythonize spare matrix constructions(?)
	# Also serialise vocab as part of the model, makes a faster __contains__ lookup
	# When applying SVD, the result is a dense matrix, change the type accordingly to not waste any memory/computing time by storing a dense matrix in sparse type
class VSMVectorizer(BaseEstimator, VectorizerMixin):
	def __init__(self, window_size, weighting='ppmi', min_frequency=0, lowercase=True, stop_words=None, encoding='utf-8',
				 max_features=None, preprocessor=None, tokenizer=None, analyzer='word', binary=False, sppmi_shift=0,
				 token_pattern=r'(?u)\b\w\w+\b', decode_error='strict', strip_accents=None, input='content',
				 ngram_range=(1, 1), cds=1., dim_reduction=None, dim_reduction_kwargs={}, random_state=1105,
				 context_window_weighting='constant', context_vector_integration=None, context_vector_integration_kwargs={},
				 word_white_list=set(), subsampling_rate=None, cache_intermediary_results=False, cache_path='~/.wort_data/model_cache',
				 log_level=logging.INFO, log_file=None, nn_eps=1.e-14):
		"""
		TODO: documentation...
		:param window_size:
		:param weighting:
		:param min_frequency:
		:param lowercase:
		:param stop_words:
		:param encoding:
		:param max_features:
		:param preprocessor:
		:param tokenizer:
		:param analyzer:
		:param binary:
		:param sppmi_shift:
		:param token_pattern:
		:param decode_error:
		:param strip_accents:
		:param input:
		:param ngram_range:
		:param random_state:
		:param cds:
		:param dim_reduction:
		:param svd_dim:
		:param svd_eig_weighting:
		:param context_window_weighting: weighting of the context window under consideration (must be either "constant", "harmonic", "distance" or "aggressive")
		:param add_context_vectors:
		:param word_white_list:
		:param subsampling_rate:
		:param cache_intermediary_results:
		:param cache_path:
		:param log_level:
		:param log_file:
		:return:
		"""

		# Support for asymmetric context windows
		if (isinstance(window_size, tuple)):
			if (len(window_size) > 1):
				self.l_window_size = window_size[0]
				self.r_window_size = window_size[1]
			else:
				self.l_window_size = window_size[0]
				self.r_window_size = window_size[0]
		else:
			self.l_window_size = window_size
			self.r_window_size = window_size

		self.weighting = weighting
		self.min_frequency = min_frequency
		self.lowercase = lowercase
		self.stop_words = stop_words
		self.encoding = encoding
		self.max_features = max_features
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.analyzer = analyzer
		self.binary = binary
		self.sppmi_shift = sppmi_shift
		self.token_pattern = token_pattern
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.input = input
		self.ngram_range = ngram_range
		self.context_window_weighting = context_window_weighting
		self.random_state = random_state
		self.cds = cds
		self.dim_reduction_kwargs = dim_reduction_kwargs
		self.dim_reduction = dim_reduction
		self.context_vector_integration = context_vector_integration
		self.context_vector_integration_kwargs = context_vector_integration_kwargs
		self.word_white_list = word_white_list
		self.subsampling_rate = subsampling_rate
		self.cache_intermediary_results = cache_intermediary_results
		if (cache_path is not None and cache_path.startswith('~')):
			cache_path = os.path.expanduser(cache_path)
		if (not os.path.exists(cache_path)):
			os.makedirs(cache_path)
		self.cache_path = cache_path
		self.nn = None
		self.nn_eps = nn_eps

		self.inverted_index_ = {}
		self.index_ = {}
		self.p_w_ = None
		self.vocab_count_ = 0
		self.token_count_ = 0
		self.M_ = None
		self.T_ = None
		self.density_ = 0.

		self.config_registry_ = ConfigRegistry(path=cache_path, min_frequency=self.min_frequency, lowercase=self.lowercase,
											   stop_words=self.stop_words, encoding= self.encoding, max_features=self.max_features,
											   preprocessor=self.preprocessor, tokenizer=self.tokenizer, analyzer=self.analyzer,
											   token_pattern=self.token_pattern, decode_error=self.decode_error,
											   strip_accents=self.strip_accents, input=self.input, ngram_range=self.ngram_range,
											   random_state=self.random_state, subsampling_rate=self.subsampling_rate,
											   wort_white_list=self.word_white_list, window_size=window_size,
											   context_window_weighting=self.context_window_weighting, binary=binary,
											   weighting=weighting, cds=cds, sppmi_shift=sppmi_shift)
		self.io_handler_ = IOHandler(cache_path=cache_path, log_file=log_file, log_level=log_level)
		self.io_handler_.setup_logging()

	def _delete_from_vocab(self, W, idx):
		W = np.delete(W, idx)

		for i in idx:
			item = self.index_[i]
			del self.inverted_index_[item]
			del self.index_[i]

		return W

	def fit_vocabulary(self, raw_documents, analyser=None):
		if (analyser is None):
			analyser = self.build_analyzer()

		n_vocab = -1
		w = array.array('i')
		white_list_idx = set()

		# Extract vocabulary
		for doc in tqdm(raw_documents):
			for feature in analyser(doc):
				idx = self.inverted_index_.get(feature, n_vocab+1)

				# Build vocab
				if (idx > n_vocab):
					n_vocab += 1
					self.inverted_index_[feature] = n_vocab
					w.append(1)
				else:
					w[idx] += 1

				# Build white_list index
				if (feature in self.word_white_list):
					white_list_idx.add(idx)

		# Vocab was used for indexing (hence, started at 0 for the first item (NOT init!)), so has to be incremented by 1
		# to reflect the true vocab count
		n_vocab += 1
		logging.info('Finished Extracting vocabulary! n_vocab={}'.format(n_vocab))

		W = np.array(w, dtype=np.uint64)
		self.index_ = dict(zip(self.inverted_index_.values(), self.inverted_index_.keys()))

		logging.info('Filtering extremes...')
		# Filter extremes
		if (self.min_frequency > 1):
			idx = np.where(W < self.min_frequency)[0]

			if (len(self.word_white_list) > 0): # Take word_white_list into account - TODO: is there a better way?
				idx = np.array(list(set(idx.tolist()) - white_list_idx))

			W = self._delete_from_vocab(W, idx)

			n_vocab -= len(idx)

		# Max Features Filter
		if (self.max_features is not None and self.max_features < n_vocab):
			idx = np.argpartition(-W, self.max_features)[self.max_features:]

			if (len(self.word_white_list) > 0): # Take word_white_list into account - TODO: is there a better way?
				idx = np.array(list(set(idx.tolist()) - white_list_idx))

			W = self._delete_from_vocab(W, idx)

			n_vocab -= len(idx)

		# Subsampling TODO: this can certainly be optimised
		token_count = W.sum()
		if (self.subsampling_rate is not None):
			rnd = np.random.RandomState(self.random_state)
			t = self.subsampling_rate * token_count

			cand_idx = np.where(W>t)[1] # idx of words exceeding threshold

			P = 1 - np.sqrt(W * (1/t)) # `word2vec` subsampling formula
			R = rnd.rand(W.shape)

			subsample_idx = np.where(R<=P)[1] # idx of filtered words

			idx = cand_idx - subsample_idx

			if (len(self.word_white_list) > 0): # Take word_white_list into account - TODO: is there a better way?
				idx -= white_list_idx

			W = self._delete_from_vocab(W, idx)

			n_vocab -= len(idx)
		logging.info('Finished Filtering extremes! n_vocab={}; n_tokens={}'.format(n_vocab, token_count))

		self.p_w_ = W / token_count
		self.vocab_count_ = n_vocab
		self.token_count_ = token_count

		# Watch out when rebuilding the index, `self.index_` needs to be built _before_ `self.inverted_index_`
		# to reflect the updated `W` array
		self.index_ = dict(zip(range(n_vocab), self.index_.values()))
		self.inverted_index_ = dict(zip(self.index_.values(), self.index_.keys()))

	def fit_cooccurrence_matrix(self, raw_documents, analyser=None):
		if (analyser is None):
			analyser = self.build_analyzer()
		# TODO: This needs optimisation
		# https://en.wikipedia.org/wiki/Feature_hashing#Feature_vectorization_using_the_hashing_trick
		# http://datascience.stackexchange.com/questions/9918/optimizing-co-occurrence-matrix-computation
		# The construction of the co-occurrence matrix can also be chunked, the size of the vocabulary
		# is known in advance (as is the number of tokens), so the construction below, which is memory heavy
		# could be chunked into several bits to ease the memory hunger of the loops a bit
		logging.info('Constructing co-occurrence matrix...')
		# Incrementally construct coo matrix (see http://www.stefanoscerra.it)
		# This can be parallelised (inverted_index is shared and immutable and the rest is just a matrix)
		rows = array.array('I') #rows = array.array('i')
		cols = array.array('I') #cols = array.array('i')
		data = array.array('I' if self.context_window_weighting == 'constant' else 'f')

		if (isinstance(self.context_window_weighting, Callable)):
			window_weighting_fn = self.context_window_weighting
		else:
			window_weighting_fn = getattr(context_weighting, '{}_window_weighting'.format(self.context_window_weighting))

		dtype = np.uint64 if self.context_window_weighting == 'constant' else np.float64
		dtype_size = 64

		# Chunking the co-occurrence matrix construction, because the 3 arrays `rows`, `cols` and `data` become huge and overflow memory
		# This is probably a pretty bad hack, but if it works, its at least possible to create (could use `psutil`(https://pypi.python.org/pypi/psutil))
		# to figure out how much memory is available and then chunk it accordingly, but thats potentially a bit overkill (+ introduces another dependency)
		# Rather fix it properly than hacking around like this...
		# On OS X/BSD this works: `os.popen('sysctl -n hw.memsize').readlines()`, on Linux, this works: `os.popen('free -m').readlines()`, ignore Windows
		chunk_size = determine_chunk_size(dtype_size=dtype_size)
		num_chunks = math.floor(self.token_count_ / chunk_size)
		processed_chunks = 1
		processed_tokens = 0
		self.M_ = sparse.lil_matrix((self.vocab_count_, self.vocab_count_), dtype=dtype)

		for doc in tqdm(raw_documents):
			buffer = array.array('i')
			for feature in analyser(doc):
				if (feature in self.inverted_index_):
					buffer.append(self.inverted_index_[feature])
					processed_tokens += 1

			# Track co-occurrences
			l = len(buffer)
			for i in range(l):
				# Backward co-occurrences
				for distance, j in enumerate(range(max(i-self.l_window_size, 0), i), 1):
					rows.append(buffer[i])
					cols.append(buffer[j])
					data.append(window_weighting_fn(-distance, self.l_window_size)) # The -distance is a bit of an ugly hack to support non-symmetric weighting

				# Forward co-occurrences
				for distance, j in enumerate(range(i+1, min(i+self.r_window_size+1, l)), 1):
					rows.append(buffer[i])
					cols.append(buffer[j])
					data.append(window_weighting_fn(distance, self.r_window_size))

			# Convert currently chunked stuff into a co-occurrence array and add it with the previously constructed co-occurrence data
			if (processed_tokens > chunk_size):
				logging.info('Chunk limit for chunk {}/{} reached, creating sparse matrix and continuing...'.format(processed_chunks, num_chunks))
				processed_tokens = 0

				data = np.array(data, dtype=np.uint8 if self.context_window_weighting == 'constant' else np.float64, copy=False)
				rows = np.array(rows, dtype=np.uint32, copy=False)
				cols = np.array(cols, dtype=np.uint32, copy=False)

				self.M_ += sparse.csr_matrix((data.astype(dtype), (rows, cols)), shape=(self.vocab_count_, self.vocab_count_)).tolil()

				rows = array.array('I') #rows = array.array('i')
				cols = array.array('I') #cols = array.array('i')
				data = array.array('I' if self.context_window_weighting == 'constant' else 'f')
				logging.info('Finished processing chunk {}/{}!'.format(processed_chunks, num_chunks))
				processed_chunks += 1

		# Add the trailing chunk to the rest
		logging.info('Numpyifying co-occurrence data...')
		data = np.array(data, dtype=np.uint8 if self.context_window_weighting == 'constant' else np.float64, copy=False)
		rows = np.array(rows, dtype=np.uint32, copy=False)
		cols = np.array(cols, dtype=np.uint32, copy=False)

		logging.info('Finalising sparse matrix...')
		self.M_ += sparse.csr_matrix((data.astype(dtype), (rows, cols)), shape=(self.vocab_count_, self.vocab_count_)).tolil()
		self.M_ = self.M_.tocsr()
		logging.info('M.shape={}'.format(self.M_.shape))

		# Apply Binarisation
		if (self.binary):
			self.M_ = self.M_.minimum(1)

	def fit_pmi_matrix(self):
		logging.info('Applying {} weight transformation...'.format(self.weighting))

		self.T_ = sparse.lil_matrix(self.M_.shape, dtype=np.float64)

		logging.info('Applying CDS...')

		# Marginals for context (with optional context distribution smoothing)
		p_c = self.p_w_ ** self.cds if self.cds != 1 else self.p_w_

		'''
		PMI is the log of the joint probability of w and c divided by the product of their marginals
		PMI = log(P(w, c) / (P(c) * P(w)))

		The joint probability can be expressed as a conditional probability via the chain rule
		P(w, c) = P(c | w) * P(w)
		P(w, c) = P(w | c) * P(c)

		Plugging this into the PMI calculation results in
		PMI = log(P(c | w) * P(w) / (P(w) * P(c)))

		This allows P(w) (or P(c), depending on how the chain rule is applied) to be eliminated
		PMI = log(P(c | w) / P(c))
		'''
		# ...apply the PMI variant (e.g. PPMI, SPPMI, PLMI or PNPMI)
		if (isinstance(self.weighting, Callable)):
			fn_feat_transformation = self.weighting
		else:
			fn_feat_transformation = getattr(feature_transformation, '{}_transformation'.format(self.weighting))

		T = fn_feat_transformation(self.M_, self.p_w_, p_c)
		logging.info('after weight option, type(PMI)={}, PMI.shape={}'.format(type(T), T.shape))

		# Apply shift
		if (self.sppmi_shift is not None and self.sppmi_shift > 0):
			logging.info('Applying shift={}...'.format(self.sppmi_shift))
			rows, cols = T.nonzero()
			data = np.full(rows.shape, self.sppmi_shift, dtype=np.float64)
			T -= sparse.csr_matrix((data, (rows, cols)), shape=T.shape, dtype=np.float64)

		logging.info('Applying the threshold [type(PMI)={}]...'.format(type(T)))
		# Apply threshold
		self.T_ = T.maximum(0)
		logging.info('PMI ALL DONE [type(self.T_)={}]'.format(type(self.T_)))

		logging.info('Returning [density={}]...'.format(len(self.T_.nonzero()[0]) / (self.T_.shape[0] * self.T_.shape[1])))
		self.density_ = len(self.T_.nonzero()[0]) / (self.T_.shape[0] * self.T_.shape[1])

	def fit_dimensionality_reduction(self):
		if (isinstance(self.dim_reduction, Callable)):
			dim_reduction_fn = self.dim_reduction
		else:
			dim_reduction_fn = getattr(dim_reduction, '{}_dim_reduction'.format(self.dim_reduction))

		return dim_reduction_fn(X=self.T_, **self.dim_reduction_kwargs)

	def fit(self, raw_documents, y=None):

		# Shameless copy/paste from Radims word2vec Tutorial, no generators matey, need multi-pass!!!
		if (raw_documents is not None):
			if (isinstance(raw_documents, GeneratorType)):
				raise TypeError('You can\'t pass a generator as the sentences argument. Try an iterator.')

		analyser = self.build_analyzer()

		##### FIT VOCABULARY
		vocab_folder = self.config_registry_.vocab_cache_folder()
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

		##### FIT CO-OCCURRENCE MATRIX
		cooc_folder = self.config_registry_.cooccurrence_matrix_folder()
		if (cooc_folder is not None and cooc_folder != ''):
			logging.info('Loading cached co-occurrence matrix resources from {}...'.format(os.path.join(self.cache_path, cooc_folder)))
			self.M_ = self.io_handler_.load_cooccurrence_matrix(cooc_folder)
			logging.info('Cache loaded!')
		else:
			logging.info('Fitting co-occurrence matrix...')
			self.fit_cooccurrence_matrix(raw_documents=raw_documents, analyser=analyser)
			logging.info('Co-occurrence matrix fitted!')

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

	def init_neighbours(self, algorithm='brute', nn_metric='cosine', num_neighbours=10):
		self.nn = NearestNeighbors(algorithm=algorithm, metric=nn_metric, n_neighbors=num_neighbours+1).fit(self.T_)

	def neighbours(self, w, return_distance=False):
		D, I = self.nn.kneighbors(self[w], return_distance=True)

		if (D[0, 0] <= self.nn_eps): # Make sure first neighbour isn't query item
			D = D[0, 1:]
			I = I[0, 1:]
		else:
			D = D[0, :-1]
			I = I[0, :-1]

		neighbour_list = list(map(lambda i: self.index_[i], I.squeeze()))

		if (return_distance):
			return D, neighbour_list

		return neighbour_list

	def neighbour_indices(self, w, return_distance=False):
		D, I = self.nn.kneighbors(self[w], return_distance=True)

		if (D[0, 0] <= self.nn_eps):  # Make sure first neighbour isn't query item
			D = D[0, 1:]
			I = I[0, 1:]
		else:
			D = D[0, :-1]
			I = I[0, :-1]

		if (return_distance):
			return D, I

		return I

	def transform(self, raw_documents, as_matrix=False, oov='zeros', composition='none'):
		'''

		:param raw_documents:
		:param as_matrix:
		:param oov: Handling of OOV entries, "ignore" doesn't return anything for an OOV item, "random", returns a random vector, "zeros" (default) returns a vector with zeros and "ones" returns a vector with ones.
		:return:
		'''
		analyser = self.build_analyzer()

		# Build OOV handler
		if (isinstance(oov, Callable)):
			oov_fn = oov
		else:
			oov_fn = getattr(oov_handler, '{}_oov_handler'.format(oov))

		# Build composer
		if (isinstance(composition, Callable)):
			mozart = composition
		else:
			mozart = getattr(vector_composition, '{}_vectors'.format(composition))

		l = []
		# Peek if a list or a string are passed
		if (isinstance(raw_documents, list)):
			for doc in raw_documents:
				d = []
				for feature in analyser(doc):
					if (feature in self):
						d.append(self[feature])
					else:
						if (oov != 'ignore'):
							d.append(oov_fn((1, self.get_vector_size()), self.T_.dtype, self.density_, self.random_state))

				# Compose and add to list
				l.append(mozart(d))

			# Convert list of lists of sparse vectors to list of sparse matrices (scipy doesn't support sparse tensors afaik)
			if (as_matrix and composition == 'none'):
				ll = []
				for l_doc in l:
					X = l_doc.pop(0)
					for x in l_doc:
						X = sparse.vstack((X, x), format='csr')
					ll.append(X)
				return ll
		else:
			for feature in analyser(raw_documents):
				if (feature in self):
					l.append(self[feature])
				else:
					if (oov != 'ignore'):
						l.append(oov_fn((1, self.get_vector_size()), self.T_.dtype, self.density_, self.random_state))

			# Compose
			l = mozart(l)

			# Convert list of sparse vectors to sparse matrix
			if (as_matrix and composition == 'none'):
				X = l.pop(0)
				for x in l:
					X = sparse.vstack((X, x), format='csr')
				return X
			elif (not as_matrix and composition != 'none'):
				l = [l]

		return l

	def fit_transform(self, raw_documents, y=None, as_matrix=False, oov='zeros'):
		self.fit(raw_documents)
		return self.transform(raw_documents, as_matrix=as_matrix, oov=oov)

	def to_dict(self):
		d = {}
		nnz_col_idx = 1 if sparse.issparse(self.T_) else 0
		for i in self.index_.keys():
			feature_dict = {}
			for col_idx in self.T_[i].nonzero()[nnz_col_idx]:
				if (self.index_[col_idx] != self.index_[i]): # Avoid self co-occurrences
					feature_dict[self.index_[col_idx]] = self.T_[i, col_idx]
			d[self.index_[i]] = feature_dict

		return d

	def get_matrix(self):
		return self.T_

	def get_index(self):
		return self.index_

	def get_inverted_index(self):
		return self.inverted_index_

	def get_vector_size(self):
		return self.T_.shape[1]

	def __getitem__(self, item):
		return self.T_[self.inverted_index_[item]]

	def __contains__(self, item):
		return item in self.inverted_index_

	@classmethod
	def load_from_file(cls, path):
		model = VSMVectorizer(window_size=0, cache_path=path)
		model.T_ = model.io_handler_.load_pmi_matrix('')
		model.index_ = model.io_handler_.load_index('')
		model.inverted_index_ = model.io_handler_.load_inverted_index('')
		model.p_w_ = model.io_handler_.load_p_w('')
		model.M_ = model.io_handler_.load_cooccurrence_matrix('')

		return model

	def save_to_file(self, path, store_cooccurrence_matrix=False):
		# If as_dict=True, call to_dict on self.T_ prior to serialisation
		# Store a few type infos in a metadata file, e.g. the type of self.T_
		# Get all params as well
		self.io_handler_.save_pmi_matrix(self.T_, sub_folder='', base_path=path)
		self.io_handler_.save_index(self.index_, sub_folder='', base_path=path)
		self.io_handler_.save_inverted_index(self.inverted_index_, sub_folder='', base_path=path)
		self.io_handler_.save_p_w(self.p_w_, sub_folder='', base_path=path)
		if (store_cooccurrence_matrix):
			self.io_handler_.save_cooccurrence_matrix(self.M_, sub_folder='', base_path=path)