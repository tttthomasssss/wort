__author__ = 'thomas'
from scipy import sparse
import numpy as np


def probability_ratio_transformation(M, p_w, p_c, **XXX):
	# Need the conditional probability P(c | w) and the marginal P(c), but need to maintain the sparsity structure of the matrix
	# Doing it this way, keeps the matrices sparse: http://stackoverflow.com/questions/3247775/how-to-elementwise-multiply-a-scipy-sparse-matrix-by-a-broadcasted-dense-1d-arra
	P_w = sparse.lil_matrix(M.shape, dtype=np.float64)
	P_c = sparse.lil_matrix(M.shape, dtype=np.float64)
	P_w.setdiag(1 / M.sum(axis=1))
	P_c.setdiag(1 / p_c)

	# TODO: THERE IS A WEIRD FAIL SOMEWHERE IN THERE!!!
	print(np.where(M.sum(axis=1)==0))
	idx = np.where(M.sum(axis=1)==0)[0]

	print('IDX={}'.format(idx))
	if (idx[0] in XXX['inverted_index']):
		print('INV IDX ITEM: {}'.format(XXX['inverted_index'][idx[0]]))
	else:
		print('{} NOT IN INVERTED INDEX!'.format(idx))
	if (idx[0] in XXX['index']):
		print('IDX ITEM: {}'.format(XXX['inverted_index'][idx[0]]))
	else:
		print('{} NOT IN INDEX!'.format(idx))
	print('P={}'.format(p_w[idx[0]]))

	'''
	(P_w * self.M_) calculates the conditional probability P(c | w) vectorised and rowwise while keeping the matrices sparse
	Multiplication by P_c (which contains the reciprocal 1 / p_c values), achieves the division by P(c)
	'''
	return (P_w * M) * P_c


def ppmi_transformation(M, p_w, p_c, **XXX):
	P = probability_ratio_transformation(M=M, p_w=p_w, p_c=p_c, **XXX)

	# Perform log on the nonzero elements of PMI
	data = np.log(P.data)
	rows, cols = P.nonzero()

	return sparse.csr_matrix((data, (rows, cols)), shape=M.shape, dtype=np.float64)


def plmi_transformation(M, p_w, p_c):
	'''
	Pointwise localised mutual information, should place less emphasis on infrequent events

	see: http://www.aclweb.org/anthology/W14-1502 and https://aclweb.org/anthology/I/I13/I13-1056.pdf
	:param M: Word-context co-occurrence matrix
	:param p_w: Probability of words
	:param p_c: Probability of contexts
	:return: PLMI transformed matrix
	'''
	PMI = ppmi_transformation(M=M, p_w=p_w, p_c=p_c)

	P_w = sparse.lil_matrix(M.shape, dtype=np.float64)
	M_rec = sparse.lil_matrix(M.shape, dtype=np.float64)
	P_w.setdiag(p_w)
	M_rec.setdiag(1 / M.sum(axis=1))

	# Calculate joint probability P(w, c) via the chain rule P(c | w) * P(w)
	P_w_c = P_w * (M_rec * M)

	return P_w_c.multiply(PMI)


def cpmi_transformation(M, p_w, p_c):
	'''
	Compressed pointwise mutual information

	see http://www.aclweb.org/anthology/P/P16/P16-3009.pdf
	:param M: Word-context co-occurrence matrix
	:param p_w: Probability of words
	:param p_c: Probability of contexts
	:return: CPMI transformed matrix
	'''
	P = probability_ratio_transformation(M=M, p_w=p_w, p_c=p_c)
	P += sparse.csr_matrix((np.full((P.nnz,), 1.), P.nonzero()), shape=M.shape, dtype=np.float64)

	# Perform log on the nonzero elements of PMI
	data = np.log(P.data)
	rows, cols = P.nonzero()

	return sparse.csr_matrix((data, (rows, cols)), shape=M.shape, dtype=np.float64)


def pnpmi_transformation(PMI, M, M_rec, p_w, *_): #TODO: Needs testing (pizza_small.txt)
	return NotImplementedError
	# Tricky one, could normalise by -log(P(w)), -log(P(c)) or -log(P(w, c)); choose the latter because it normalises the upper & the lower bound,
	# and is nicer implementationwise (see Bouma 2009: https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf)
	#P_w_c.data = 1 / -np.log(P_w_c.data)
	#return P_w_c.multiply(PMI)


def ttest_transformation(PMI, M, M_rec, p_w, p_c): # http://www.cl.cam.ac.uk/~tp366/papers/eacl2014-polajnarclark.pdf
	raise NotImplementedError # TODO