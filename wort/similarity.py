import math

from scipy import sparse
from scipy.spatial import distance
from scipy.stats import entropy
import numpy as np

def _check_xy(x, y):

	# Sparse check
	x_prime = x.A if sparse.issparse(x) else x  # Quickhack for sparse matrices
	y_prime = y.A if sparse.issparse(y) else y

	# ndim check
	x_prime = x_prime.squeeze() if x_prime.ndim > 1 else x_prime
	y_prime = y_prime.squeeze() if y_prime.ndim > 1 else y_prime

	return x_prime, y_prime


def cosine(x, y):
	x, y = _check_xy(x, y)

	return 1 - distance.cosine(x, y)


def lin(x, y):
	'''
	Calculate the Lin distance between two numpy vectors.

	The formula is on p. 770 of Lin (1998) - Automatic Retrieval and Clustering of Similar Words
		http://aclweb.org/anthology/P/P98/P98-2127.pdf

	It basically takes the sum of the PPMI scores of the intersection of the two vectors, divided by
	the their union and is somewhat Jaccard-y.

	:param x: vector 1, numpy array
	:param y: vector 2, numpy array
	:return: lin similarity between vector 1 and vector 2, float
	'''
	x, y = _check_xy(x, y)

	idx = np.intersect1d(np.where(x>0), np.where(y>0))

	enum = np.sum(x[idx] + y[idx])
	denom = x.sum() + y.sum()

	return (enum / denom)


def jensen_shannon(x, y, square_root=False):
	x, y = _check_xy(x, y)

	m = 0.5 * (x + y)

	x_m = entropy(x, m, base=2) # `entropy(...)` calculates the KL-divergence if `qk` is not None
	y_m = entropy(y, m, base=2)

	sim = 1 - (0.5 * (x_m + y_m))

	return np.sqrt(sim) if square_root else sim


def weeds_precision(x, y):
	x, y = _check_xy(x, y)

	idx = np.intersect1d(np.where(x > 0), np.where(y > 0))

	enum = x[idx].sum()
	denom = x.sum()

	return (enum / denom)

def weeds_recall(x, y):
	x, y = _check_xy(x, y)

	idx = np.intersect1d(np.where(x > 0), np.where(y > 0))

	enum = y[idx].sum()
	denom = y.sum()

	return (enum / denom)


def weeds_f1(x, y):
	x, y = _check_xy(x, y)

	prec = weeds_precision(x, y)
	rec = weeds_recall(x, y)

	f1 = 2 * ((rec * prec) / (rec + prec))

	return f1


def binc(x, y):
	x, y = _check_xy(x, y)

	lin_sim = lin(x, y)
	weeds_prec = weeds_precision(x, y)

	return math.sqrt(lin_sim * weeds_prec)


def alpha_skew(x, y, alpha=0.99):
	x, y = _check_xy(x, y)

	y = (alpha * y) + ((1 - alpha) * x)

	return 1 - entropy(x, y, base=2)


def weeds_cosine(x, y):
	x, y = _check_xy(x, y)

	cos = cosine(x, y)
	weeds_prec = weeds_precision(x, y)

	return math.sqrt(cos * weeds_prec)


def clarke_inclusion(x, y):
	x, y = _check_xy(x, y)

	return np.minimum(x, y).sum() / x.sum()


def inverse_clarke_inclusion(x, y):
	x, y = _check_xy(x, y)

	cde = clarke_inclusion(x, y)

	return math.sqrt(cde * (1 - cde))


def slqs(x, y):
	# BE CAREFUL, x and y ARE NOT THE WORD REPRESENTATIONS BUT SHOULD BE THE MEDIAN ENTROPIES OF THE `N`
	# LARGEST CONTEXTS OF x AND y.
	x, y = _check_xy(x, y)

	return 1 - (x / y)