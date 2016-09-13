from scipy import sparse


def none_vectors(l): # Identity function
	return l


def add_vectors(l):
	x0 = l[0].tolil()
	for x in l[1:]:
		x0 += x.tolil()

	return x0.tocsr()


def multiply_vectors(l):
	x0 = l[0]
	for x in l[1:]:
		x0 = x0.multiply(x)

	return x0


def min_vectors(l):
	X = l[0]
	for x in l[1:]:
		X = sparse.vstack((X, x), format='csr')

	return X.min(axis=0).tocsr()


def max_vectors(l):
	X = l[0]
	for x in l[1:]:
		X = sparse.vstack((X, x), format='csr')

	return X.max(axis=0).tocsr()