__author__ = 'thomas'
from scipy import sparse
import numpy as np


def random_oov_handler(shape, dtype, density, random_state):
		return sparse.random(shape[0], shape[1], density=density, format='csr', random_state=random_state, dtype=dtype)


def ignore_oov_handler(*_):
	pass


def zeros_oov_handler(shape, dtype, *_):
	return sparse.csr_matrix(np.zeros(shape), dtype=dtype)


def ones_oov_handler(shape, dtype, *_):
	return sparse.csr_matrix(np.ones(shape), dtype=dtype)