__author__ = 'thomas'
import logging

from scipy import sparse
from sparsesvd import sparsesvd
import numpy as np


def svd_dim_reduction(X, **kwargs):
	dim = kwargs['dimensionality'] # Fail loudly if not provided
	eig_weighting = kwargs.pop('eig_weighting', 1.)

	logging.info('Applying SVD with dimensionality={}...'.format(dim))
	Ut, S, Vt = sparsesvd(X.tocsc() if sparse.issparse(X) else sparse.csc_matrix(X), dim)

	# Perform Context Weighting
	S = sparse.csr_matrix(np.diag(S ** eig_weighting))

	W = sparse.csr_matrix(Ut.T).dot(S) # Word vectors
	C = sparse.csr_matrix(Vt.T).dot(S) # Context vectors

	return W, C