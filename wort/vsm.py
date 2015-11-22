__author__ = 'thomas'
from types import GeneratorType
import array
import os

from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin
from sparsesvd import sparsesvd
import numpy as np

# TODO: SVD based on http://www.aclweb.org/anthology/Q/Q15/Q15-1016.pdf, esp. chapter 7, practical recommendations
	# Context Window Weighting
	# Subsampling
	# Normalisation
	# Hellinger PCA
	# NMF as an alternative to SVD
	# Support min_df and max_df
	# Proper Logging would be nice
	# Optimise the shizzle-whizzle
	# Remove memmap? properly support it?
	# Improve numerical precision
class VSMVectorizer(BaseEstimator, VectorizerMixin):
	def __init__(self, window_size, weighting='ppmi', min_frequency=0, lowercase=True, stop_words=None, encoding='utf-8',
				 max_features=None, preprocessor=None, tokenizer=None, analyzer='word', binary=False, sppmi_shift=1,
				 token_pattern=r'(?u)\b\w\w+\b', decode_error='strict', strip_accents=None, input='content',
				 ngram_range=(1, 1), cds=1, svd_dim=None, svd_eig_weighting=1, add_context_vectors=True,
				 use_memmap=False, memmap_path=None):

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
		self.use_memmap = use_memmap
		self.memmap_path = memmap_path
		self.input = input
		self.ngram_range = ngram_range
		self.cds = cds
		self.svd_dim = svd_dim
		self.svd_eig_weighting = svd_eig_weighting
		self.add_context_vectors = add_context_vectors

		self.inverted_index_ = {}
		self.index_ = {}
		self.p_w_ = None
		self.vocab_count_ = 0
		self.M_ = None
		self.T_ = None

	def _delete_from_vocab(self, W, idx):
		W = np.delete(W, idx)

		for i in idx:
			item = self.index_[i]
			del self.inverted_index_[item]
			del self.index_[i]

		return W

	def _construct_cooccurrence_matrix(self, raw_documents):
		analyser = self.build_analyzer()

		n_vocab = -1
		w = array.array('i')

		# Extract vocabulary
		print ('Extracting vocabulary...')
		for doc in raw_documents:
			for feature in analyser(doc):
				idx = self.inverted_index_.get(feature, n_vocab + 1)

				# Build vocab
				if (idx > n_vocab):
					n_vocab += 1
					self.inverted_index_[feature] = n_vocab
					w.append(1)
				else:
					w[idx] += 1

		# Vocab was used for indexing (hence, started at 0 for the first item (NOT init!)), so has to be incremented by 1
		# to reflect the true vocab count
		n_vocab += 1

		W = np.array(w, dtype=np.uint32)
		self.index_ = dict(zip(self.inverted_index_.values(), self.inverted_index_.keys()))

		print('Filtering extremes...')
		# Filter extremes
		if (not self.binary and self.min_frequency > 0):
			idx = np.where(W < self.min_frequency)[0]
			W = self._delete_from_vocab(W, idx)

			n_vocab -= len(idx)

		# Max Features Filter
		if (self.max_features is not None and self.max_features < n_vocab):
			idx = np.argpartition(-W)[self.max_features + 1:]
			W = self._delete_from_vocab(W, idx)

			n_vocab -= len(idx)

		self.p_w_ = W / W.sum()
		self.vocab_count_ = n_vocab
		self.inverted_index_ = dict(zip(self.inverted_index_.keys(), range(n_vocab)))

		print('Constructing co-occurrence matrix...')
		# Incrementally construct coo matrix (see http://www.stefanoscerra.it)
		# This can be parallelised (inverted_index is shared and immutable and the rest is just a matrix)
		rows = array.array('i')
		cols = array.array('i')
		data = array.array('i')

		for doc in raw_documents:
			buffer = array.array('i')
			for feature in analyser(doc):
				if (feature in self.inverted_index_):
					buffer.append(self.inverted_index_[feature])

			# Track co-occurrences
			l = len(buffer)
			for i in range(l):
				# Backward co-occurrences
				for j in range(max(i-self.window_size, 0), i):
					rows.append(buffer[i])
					cols.append(buffer[j])
					data.append(1)

				# Forward co-occurrences
				for j in range(i+1, min(i + self.window_size + 1, l)):
					rows.append(buffer[i])
					cols.append(buffer[j])
					data.append(1)

		print('Creating sparse matrix...')
		data = np.array(data, dtype=np.uint32, copy=False)
		rows = np.array(rows, dtype=np.uint32, copy=False)
		cols = np.array(cols, dtype=np.uint32, copy=False)

		self.M_ = sparse.coo_matrix((data, (rows, cols)))

		# Apply Binarisation
		if (self.binary):
			self.M_ = np.minimum(self.M_, 1)

	'''
	def _construct_cooccurrence_matrix_mmap(self):
		self.M_ = np.memmap(os.path.join(self.memmap_path, 'M.dat'), dtype=np.uint32, mode='w+', shape=(self.vocab_count_, self.vocab_count_))
		self.M_[:] = np.zeros((self.vocab_count_, self.vocab_count_))

		for idx in self.index_.keys():
			occurrences = np.where(self.temp_tokens_store_==idx)[0]
			for occ_idx in occurrences:
				cooccurrence_window = np.concatenate((self.temp_tokens_store_[occ_idx - self.window_size:occ_idx], self.temp_tokens_store_[occ_idx + 1:occ_idx + 1 + self.window_size]))

				self.M_[idx, cooccurrence_window] += 1

		self.M_.flush()

	def _construct_cooccurrence_matrix(self, raw_documents):
		self.M_ = sparse.dok_matrix((self.vocab_count_, self.vocab_count_), dtype=np.uint32)

		#for doc in raw_documents:
		#	for feature in self.analyser_fn_(doc):


		# To mem heavy
		for idx in self.index_.keys():
			occurrences = np.where(self.temp_tokens_store_==idx)[0]
			for occ_idx in occurrences:
				cooccurrence_window = np.concatenate((self.temp_tokens_store_[occ_idx - self.window_size:occ_idx], self.temp_tokens_store_[occ_idx + 1:occ_idx + 1 + self.window_size]))

				self.M_[idx, cooccurrence_window] += 1
	'''

	def _apply_weight_option(self, PMI, P_w_c, p_c):
		if (self.weighting == 'ppmi'):
			return PMI
		elif (self.weighting == 'plmi'):
			return P_w_c * PMI
		elif (self.weighting == 'pnpmi'):
			raise NotImplementedError #(P_w_c[idx, row] * (1 / -(np.log(p_c)))) * pmi # TODO
		elif (self.weighting == 'sppmi'):
			return PMI - np.log(self.sppmi_shift)

	def _transform_memmap(self):
		self.T_ = np.memmap(os.path.join(self.memmap_path, '%s.dat' % (self.weighting.upper(),)), dtype=np.uint32, mode='w+', shape=(self.vocab_count_, self.vocab_count_))
		self.T_[:] = np.zeros((self.vocab_count_, self.vocab_count_), dtype=np.float64)

		# Joint Probability for all co-occurrences, P(w, c) = P(c | w) * P(w) = P(w | c) * P(c)
		P_w_c = self.M_ / self.M_.sum(axis=1) * self.p_w_.reshape(-1, 1)

		# Perform weighting
		for idx in self.index_.keys():
			row = np.where(self.M_[idx, :] > 0)[0]

			# Marginals for context
			p_c = self.p_w_[row] ** self.cds

			# PMI
			pmi = np.log(P_w_c[idx, row]) - (np.log(self.p_w_[idx]) + np.log(p_c))

			# Apply PMI variant (e.g. PPMI, SPPMI, PLMI or PNPMI)
			tpmi = self._apply_weight_option(pmi, self.weighting, p_c, idx, row)

			self.T_[idx, row] = np.maximum(0, tpmi)

		return self.T_

	def _weight_transformation(self):
		self.T_ = sparse.lil_matrix(self.M_.shape, dtype=np.float64)

		# Joint Probability for all co-occurrences, P(w, c) = P(c | w) * P(w) = P(w | c) * P(c)
		# Doing it this way, keeps P_w_c a sparse matrix: http://stackoverflow.com/questions/3247775/how-to-elementwise-multiply-a-scipy-sparse-matrix-by-a-broadcasted-dense-1d-arra
		P_w = sparse.lil_matrix(self.M_.shape)
		P_w.setdiag((self.p_w_.reshape(-1, 1) / self.M_.sum(axis=1)))
		P_w_c = P_w * self.M_

		# Marginals for context (with optional context distribution smoothing)
		p_c = self.p_w_ ** self.cds

		# The product of all P(w) and P(c) marginals is the outer product of p_w and p_c
		P_wc_marginals = np.outer(self.p_w_, p_c)

		# PMI matrix is then the log difference between the joints and the marginals
		P_w_c.data = np.log(P_w_c.data)
		P_wc_marginals.data = np.log(P_wc_marginals.data)
		PMI = P_w_c - P_wc_marginals

		# Apply PMI variant (e.g. PPMI, SPPMI, PLMI or PNPMI) and threshold at 0
		self.T_ = np.maximum(0, self._apply_weight_option(PMI, P_w_c, p_c))

		# Apply SVD
		if (self.svd_dim is not None):
			Ut, S, Vt = sparsesvd(self.T_.tocsc(), self.svd_dim)

			# Perform Context Weighting
			S = sparse.csr_matrix(np.diag(S ** self.svd_eig_weighting))

			W = sparse.csr_matrix(Ut.T).dot(S)
			V = sparse.csr_matrix(Vt.T).dot(S)

			# Add context vectors
			if (self.add_context_vectors):
				self.T_ = W + V
			else:
				self.T_ = W

		return self.T_

	def fit(self, raw_documents, y=None):

		# Shameless copy/paste from Radims word2vec Tutorial, no generators matey, need multi-pass!!!
		if raw_documents is not None:
			if isinstance(raw_documents, GeneratorType):
				raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")

		self._construct_cooccurrence_matrix(raw_documents)

		# Construct Co-Occurrence Matrix
		#if (self.use_memmap):
		#	self._construct_cooccurrence_matrix_mmap()
		#else:
		#	self._construct_cooccurrence_matrix(raw_documents)

		# Apply weighting transformation
		self._weight_transformation()

		return self

	def transform(self, raw_documents):
		# todo move to a different class or rename?
		# Apply the weighting transformation
		#if (self.use_memmap):
		#	return sparse.csr_matrix(self._transform_memmap())
		#else:
		#	return self._transform()
		if (self.T_ is not None):
			return self.T_
		else: # TODO: perform lookup
			raise NotImplementedError

	def fit_transform(self, raw_documents, y=None):
		self.fit(raw_documents)
		return self.transform(raw_documents)

