__author__ = 'thomas'
from types import GeneratorType
import array
import logging
import math
import os
import sys

from scipy import sparse
from scipy.special import expit as sigmoid
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin
from sparsesvd import sparsesvd
from tqdm import *
import joblib
import numpy as np

from wort import utils

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
				 max_features=None, preprocessor=None, tokenizer=None, analyzer='word', binary=False, sppmi_shift=1,
				 token_pattern=r'(?u)\b\w\w+\b', decode_error='strict', strip_accents=None, input='content',
				 ngram_range=(1, 1), cds=1., dim_reduction=None, svd_dim=None, svd_eig_weighting=1, random_state=1105,
				 context_window_weighting='constant', add_context_vectors=True, word_white_list=set(),
				 subsampling_rate=None,cache_intermediary_results=False, cache_path=None, log_level=logging.INFO,
				 log_file=None):
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

		self.window_size = window_size
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
		self.svd_dim = svd_dim
		self.svd_eig_weighting = svd_eig_weighting
		self.dim_reduction = dim_reduction
		self.add_context_vectors = add_context_vectors
		self.word_white_list = word_white_list
		self.subsampling_rate = subsampling_rate
		self.cache_intermediary_results = cache_intermediary_results
		self.cache_path = cache_path

		self.inverted_index_ = {}
		self.index_ = {}
		self.p_w_ = None
		self.vocab_count_ = 0
		self.M_ = None
		self.T_ = None

		self.log_level_ = log_level
		self.log_file_ = log_file
		self._setup_logging()

	def _setup_logging(self):
		log_formatter = logging.Formatter(fmt='%(asctime)s: %(levelname)s - %(message)s', datefmt='[%d/%m/%Y %H:%M:%S %p]')
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

	def _delete_from_vocab(self, W, idx):
		W = np.delete(W, idx)

		for i in idx:
			item = self.index_[i]
			del self.inverted_index_[item]
			del self.index_[i]

		return W

	def _constant_window_weighting(self, _):
		return 1

	def _aggressive_window_weighting(self, distance):
		return 2 ** (1 - abs(distance))

	def _very_aggressive_window_weighting(self, distance):
		return 2 ** (1 - (distance ** 2))

	def _harmonic_window_weighting(self, distance):
		return 1. / abs(distance) # Thats what GloVe is doing

	def _distance_window_weighting(self, distance):
		return (self.window_size - abs(distance)) / self.window_size # Thats what word2vec is doing

	def _sigmoid_window_weighting(self, distance):
		return sigmoid(distance)

	def _inverse_sigmoid_window_weighting(self, distance):
		return 1 - sigmoid(distance)

	def _absolute_sigmoid_window_weighting(self, distance):
		return abs(sigmoid(distance))

	def _absolut_inverse_sigmoid_window_weighting(self, distance):
		return 1 - abs(sigmoid(distance))

	def _construct_cooccurrence_matrix(self, raw_documents):
		analyser = self.build_analyzer()

		n_vocab = -1
		w = array.array('i')
		white_list_idx = set()

		# Extract vocabulary
		logging.info('Extracting vocabulary...')
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

		logging.info('Finished Filtering extremes! n_vocab={}'.format(n_vocab))

		self.p_w_ = W / token_count
		self.vocab_count_ = n_vocab

		# Watch out when rebuilding the index, `self.index_` needs to be built _before_ `self.inverted_index_`
		# to reflect the updated `W` array
		self.index_ = dict(zip(range(n_vocab), self.index_.values()))
		self.inverted_index_ = dict(zip(self.index_.values(), self.index_.keys()))

		logging.info('Constructing co-occurrence matrix...')
		# Incrementally construct coo matrix (see http://www.stefanoscerra.it)
		# This can be parallelised (inverted_index is shared and immutable and the rest is just a matrix)
		rows = array.array('I') #rows = array.array('i')
		cols = array.array('I') #cols = array.array('i')
		data = array.array('I' if self.context_window_weighting == 'constant' else 'f')

		window_weighting_fn = getattr(self, '_{}_window_weighting'.format(self.context_window_weighting))

		for doc in tqdm(raw_documents):
			buffer = array.array('i')
			for feature in analyser(doc):
				if (feature in self.inverted_index_):
					buffer.append(self.inverted_index_[feature])

			# Track co-occurrences
			l = len(buffer)
			for i in range(l):
				# Backward co-occurrences
				#logging.info('BACKWARD RANGE: {}'.format(list(range(max(i-self.window_size, 0), i))))
				for distance, j in enumerate(range(max(i-self.window_size, 0), i), 1):
					rows.append(buffer[i])
					cols.append(buffer[j])
					data.append(window_weighting_fn(-distance)) # The -distance is a bit of an ugly hack to support non-symmetric weighting
					#logging.info('BWD DISTANCE: {}; WORD={}'.format(distance, self.index_[buffer[j]]))

				# Forward co-occurrences
				#logging.info('FORWARD RANGE: {}'.format(list(range(i+1, min(i+self.window_size+1, l)))))
				for distance, j in enumerate(range(i+1, min(i+self.window_size+1, l)), 1):
					rows.append(buffer[i])
					cols.append(buffer[j])
					data.append(window_weighting_fn(distance))
					#logging.info('FWD DISTANCE: {}; WORD={}'.format(distance, self.index_[buffer[j]]))

		# TODO: This is still a bit of a bottleneck
		#		Either cythonize the shit
		#		Or chunk it up and create several sparse arrays that get added (?)
		logging.info('Numpyifying co-occurrence data...')
		data = np.array(data, dtype=np.uint64 if self.context_window_weighting == 'constant' else np.float64, copy=False)
		rows = np.array(rows, dtype=np.uint32, copy=False)
		cols = np.array(cols, dtype=np.uint32, copy=False)

		#if (self.cache_intermediary_results):
		#	logging.info('Storing raw array co-occurrence data...')
		#	utils.numpy_to_hdf(data, self.cache_path, 'data.cooc')
		#	utils.numpy_to_hdf(rows, self.cache_path, 'rows.cooc')
		#	utils.numpy_to_hdf(cols, self.cache_path, 'cols.cooc')

		logging.info('Creating sparse matrix...')
		# Create a csr_matrix straight away!!!
		#self.M_ = sparse.coo_matrix((data, (rows, cols)), dtype=np.uint64 if self.context_window_weighting == 'constant' else np.float64).tocsr() # Scipy seems to not handle numeric overflow in a very graceful manner
		dtype = np.uint64 if self.context_window_weighting == 'constant' else np.float64
		self.M_ = sparse.csr_matrix((data), (rows, cols), dtype=dtype, shape=(n_vocab, n_vocab))
		logging.info('M.shape={}'.format(self.M_.shape))

		# Apply Binarisation
		if (self.binary):
			self.M_ = self.M_.minimum(1)

	def _apply_weight_option(self, PMI, P_w_c, p_c):
		# TODO: re-check results for `plmi` and `pnpmi`
		if (self.weighting == 'ppmi'):
			return PMI
		elif (self.weighting == 'plmi'):
			return P_w_c.multiply(PMI)
		elif (self.weighting == 'pnpmi'):
			# Tricky one, could normalise by -log(P(w)), -log(P(c)) or -log(P(w, c)); choose the latter because it normalises the upper & the lower bound,
			# and is nicer implementationwise (see Bouma 2009: https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf)
			P_w_c.data = 1 / -np.log(P_w_c.data)
			return P_w_c.multiply(PMI)

	def _weight_transformation(self):
		logging.info('Applying {} weight transformation...'.format(self.weighting))

		self.T_ = sparse.lil_matrix(self.M_.shape, dtype=np.float64)

		# Joint Probability for all co-occurrences, P(w, c) = P(c | w) * P(w) = P(w | c) * P(c)
		# Doing it this way, keeps P_w_c a sparse matrix: http://stackoverflow.com/questions/3247775/how-to-elementwise-multiply-a-scipy-sparse-matrix-by-a-broadcasted-dense-1d-arra
		P_w = sparse.lil_matrix(self.M_.shape, dtype=np.float64)
		logging.info('M sum max={}'.format(np.amax(self.M_.sum(axis=1))))
		logging.info('M sum={}'.format(self.M_.sum(axis=1).shape)) # TODO: Shit goes wrong here!!!!!!!!
		logging.info('M_dtype={}; M_sum_dtype={}'.format(self.M_.dtype, self.M_.sum(axis=1).dtype))
		logging.info('New p_w_ shape={}; M.sum(axis=1) shape={}'.format(self.p_w_.reshape(-1, 1).shape, self.M_.sum(axis=1).shape)) # TODO: THE ERROR IS IN ONE OF THE TWO CALLS!!!!!!!!
		P_w.setdiag((self.p_w_.reshape(-1, 1) / self.M_.sum(axis=1)))

		logging.info('Calculating Joints...') #TODO: check if this is correct!

		P_w_c = P_w * self.M_

		logging.info('Applying CDS...')

		# Marginals for context (with optional context distribution smoothing)
		p_c = self.p_w_ ** self.cds if self.cds != 1 else self.p_w_

		logging.info('Calculating Marginals...')


		# The product of all P(w) and P(c) marginals is the outer product of p_w and p_c
		P_wc_marginals = np.outer(self.p_w_, p_c) # TODO: Cythonize?

		logging.info('Taking logs...')

		logging.info('Construction the COO PMI matrix')
		# Construct another CSR matrix as we go and ...
		data = np.log(P_w_c.data / P_wc_marginals[P_w_c.nonzero()]) # Doing the division first maintains the sparsity of the matrix
		rows, cols = P_w_c.nonzero()

		logging.info('Applying the PMI option')
		# ...apply the PMI variant (e.g. PPMI, SPPMI, PLMI or PNPMI)
		PMI = self._apply_weight_option(sparse.csr_matrix((data, (rows, cols)), shape=self.M_.shape, dtype=np.float64), P_w_c, p_c)

		# Apply shift
		if (self.sppmi_shift is not None and self.sppmi_shift > 0):
			PMI.data -= math.log(self.sppmi_shift) # Maintain sparsity structure!

		logging.info('Applying the threshold [type(PMI)={}]...'.format(type(PMI)))
		# Apply threshold
		self.T_ = PMI.maximum(0)
		logging.info('PMI ALL DONE [type(self.T_)={}]'.format(type(self.T_)))

		# Apply SVD
		if (self.dim_reduction == 'svd'):
			logging.info('Applying SVD...')
			Ut, S, Vt = sparsesvd(self.T_.tocsc() if sparse.issparse(self.T_) else sparse.csc_matrix(self.T_), self.svd_dim)

			# Perform Context Weighting
			S = sparse.csr_matrix(np.diag(S ** self.svd_eig_weighting))

			W = sparse.csr_matrix(Ut.T).dot(S)
			V = sparse.csr_matrix(Vt.T).dot(S)

			# Add context vectors
			if (self.add_context_vectors):
				self.T_ = W + V
			else:
				self.T_ = W

		logging.info('Returning [density={}]...'.format(len(self.T_.nonzero()[0]) / (self.T_.shape[0] * self.T_.shape[1])))

		return self.T_

	def fit(self, raw_documents, y=None):

		# Shameless copy/paste from Radims word2vec Tutorial, no generators matey, need multi-pass!!!
		if (raw_documents is not None):
			if (isinstance(raw_documents, GeneratorType)):
				raise TypeError('You can\'t pass a generator as the sentences argument. Try an iterator.')

		if (not os.path.exists(os.path.join(self.cache_path, 'M.hdf'))):
			logging.info('No cache available at {}! Constructing the co-occurrence matrix!'.format(self.cache_path))
			self._construct_cooccurrence_matrix(raw_documents)

			if (self.cache_intermediary_results):
				# Store the matrix bits and pieces in several different files due to performance and size
				logging.info('Caching co-occurrence matrix to path: {}...'.format(self.cache_path))
				utils.sparse_matrix_to_hdf(self.M_, self.cache_path, 'M.hdf')
				logging.info('Finished caching co-occurence matrix!')

				logging.info('Caching word probability distribution to path: {}...'.format(os.path.join(self.cache_path, 'p_w.joblib')))
				joblib.dump(self.p_w_, os.path.join(self.cache_path, 'p_w.joblib'), compress=3)
				logging.info('Finished caching word probability distribution!')

				logging.info('Caching index to path: {}...'.format(os.path.join(self.cache_path, 'index.joblib')))
				joblib.dump(self.index_, os.path.join(self.cache_path, 'index.joblib'), compress=3)
				logging.info('Finished caching index!')

				logging.info('Caching inverted index to path: {}...'.format(os.path.join(self.cache_path, 'inverted_index.joblib')))
				joblib.dump(self.inverted_index_, os.path.join(self.cache_path, 'inverted_index.joblib'), compress=3)
				logging.info('Finished caching inverted index!')

			# Apply weighting transformation
			self._weight_transformation()
		else:
			logging.info('Found cached co-occurrence matrix at {}! Applying {} from cache!'.format(self.cache_path, self.weighting))
			self.weight_transformation_from_cache()

		return self

	def weight_transformation_from_cache(self):
		#self.M_ = joblib.load(os.path.join(self.cache_path, 'M_cooccurrence.joblib'))
		self.M_ = utils.hdf_to_sparse_csx_matrix(self.cache_path, 'M.hdf', sparse_format='csr')
		self.p_w_ = joblib.load(os.path.join(self.cache_path, 'p_w.joblib'))
		self.index_ = joblib.load(os.path.join(self.cache_path, 'index.joblib'))
		self.inverted_index_ = joblib.load(os.path.join(self.cache_path, 'inverted_index.joblib'))

		self._weight_transformation()

		return self

	def transform(self, raw_documents):
		# todo move to a different class or rename?
		if (self.T_ is not None):
			return self.T_
		else: # TODO: perform lookup
			raise NotImplementedError

	def fit_transform(self, raw_documents, y=None):
		self.fit(raw_documents)
		return self.transform(raw_documents)

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
		return self.T_[self.inverted_index_[item]].A

	def __contains__(self, item):
		return item in self.inverted_index_

	@classmethod
	def load_from_file(cls, path, as_dict=False):
		model = VSMVectorizer(window_size=5)

		logging.info('Loading hdf file...')
		model.T_ = utils.hdf_to_sparse_csx_matrix(path, 'T.hdf', sparse_format='csr') # TODO: static hack on sparse_format!!!
		logging.info('Loading rest...')
		model.index_ = joblib.load(os.path.join(path, 'index.joblib'))
		model.inverted_index_ = joblib.load(os.path.join(path, 'inverted_index.joblib'))
		model.p_w_ = joblib.load(os.path.join(path, 'p_w.joblib'))
		logging.info('Everything loaded!')

		return model

	def save_to_file(self, path, as_dict=False):
		# If as_dict=True, call to_dict on self.T_ prior to serialisation
		# Store a few type infos in a metadata file, e.g. the type of self.T_
		# Get all params as well
		utils.sparse_csx_matrix_to_hdf(self.T_, path, 'T.hdf')
		joblib.dump(self.index_, os.path.join(path, 'index.joblib'), compress=3)
		joblib.dump(self.inverted_index_, os.path.join(path, 'inverted_index.joblib'), compress=3)
		joblib.dump(self.p_w_, os.path.join(path, 'p_w.joblib'), compress=3)