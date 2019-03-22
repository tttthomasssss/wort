__author__ = 'thomas'
import logging

from scipy import sparse
from sklearn.decomposition import NMF
from sparsesvd import sparsesvd
import numpy as np


def svd_dim_reduction(X, **kwargs):
	dim = kwargs['dimensionality'] # Fail loudly if not provided
	eig_weighting = kwargs.pop('eig_weighting', 1.)
	skip_top_n_dimensions = kwargs.pop('skip_top_n_dimensions', 0)

	logging.info('Applying SVD with dimensionality={}...'.format(dim))
	Ut, s, Vt = sparsesvd(X.tocsc() if sparse.issparse(X) else sparse.csc_matrix(X), dim)

	# Perform Context Weighting
	S = np.diag(s ** eig_weighting)

	W = Ut.T.dot(S)[:, skip_top_n_dimensions:] # Word vectors
	C = Vt.T.dot(S)[:, skip_top_n_dimensions:] # Context vectors

	return W, C


def nmf_dim_reduction(X, **kwargs):
	dim = kwargs.pop('dimensionality') if 'dimensionality' in kwargs else kwargs.pop('n_components') # must be one of the two, latter is sklearn NMF lingo
	skip_top_n_dimensions = kwargs.pop('skip_top_n_dimensions', 0)

	logging.info('Applying NMF with dimensionality={}...'.format(dim))
	nmf = NMF(n_components=dim, **kwargs)

	W = nmf.fit_transform(X)
	C = nmf.components_

	W = W[:, skip_top_n_dimensions:]
	C = C[:, skip_top_n_dimensions:]

	return W, C

# Random Indexing: https://www.diva-portal.org/smash/get/diva2:1041127/FULLTEXT01.pdf
