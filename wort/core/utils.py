__author__ = 'thk22'
# Code more or less shamelessly stolen from: http://stackoverflow.com/questions/11129429/storing-numpy-sparse-matrix-in-hdf5-pytables/11130235#11130235
import os
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy import sparse
import numpy as np
import tables


class LemmaTokenizer(object):

	def __init__(self):
		self.wnl = WordNetLemmatizer()

	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# TODO: The file appending thing is a bit ugly, should add some custom naming facilities...
# TODO: Support compression for all HDF operations, see http://stackoverflow.com/questions/20118560/compressing-array-with-pytables#answer-20139553 and http://www.pytables.org/usersguide/libref/helper_classes.html#filtersclassdescr


def numpy_to_hdf(obj, path, name):
	with tables.open_file(os.path.join(path, '{}.hdf'.format(name)), 'w') as f:
		atom = tables.Atom.from_dtype(obj.dtype)
		arr = f.create_carray(f.root, name, atom, obj.shape)
		arr[:] = obj


def hdf_to_numpy(path, name, compression_level=0, compression_lib='zlib'):
	filters = tables.Filters(complevel=compression_level, complib=compression_lib)
	with tables.open_file(os.path.join(path, '{}.hdf'.format(name)), 'r', filters=filters) as f:
		try: # TODO: QUICKHACK - REMOVE LATER!!!!!
			arr = np.array(getattr(f.root, name).read())
		except tables.exceptions.NoSuchNodeError:
			arr = np.array(getattr(f.root, name.split('.')[0]).read())
	return arr


def sparse_matrix_to_hdf(obj, path, name):
	if (sparse.isspmatrix_csr(obj) or sparse.isspmatrix_csc(obj)):
		sparse_csx_matrix_to_hdf(obj, path, name)
	elif (sparse.isspmatrix_coo(obj)):
		sparse_coo_matrix_to_hdf(obj, path, name)
	else:
		raise ValueError('Type {} not yet supported for serialisation!'.format(type(obj)))


def hdf_to_sparse_matrix(path, name, sparse_format):
	if (sparse_format in ['csr', 'csc']):
		return hdf_to_sparse_csx_matrix(path, name, sparse_format)
	elif (sparse_format == 'coo'):
		return hdf_to_sparse_coo_matrix(path, name)
	else:
		raise ValueError('Sparse format "{}" not yet supported for de-serialisation!'.format(sparse_format))


def hdf_to_sparse_csx_matrix(path, name, sparse_format):
	attrs = _get_attrs_from_hdf_file(path, name, 'csx', ['data', 'indices', 'indptr', 'shape'])
	constructor = getattr(sparse, '{}_matrix'.format(sparse_format))

	return constructor(tuple(attrs[:3]), shape=tuple(attrs[3]))


def hdf_to_sparse_coo_matrix(path, name):
	attrs = _get_attrs_from_hdf_file(path, name, 'coo', ['data', 'rows', 'cols', 'shape'])

	return sparse.coo_matrix((attrs[0], tuple(attrs[1:3])), shape=attrs[3])


def _get_attrs_from_hdf_file(path, name, sparse_format, attributes):
	with tables.open_file(os.path.join(path, name), 'r') as f:
		attrs = []
		for attr in attributes:
			attrs.append(getattr(f.root, '{}_{}'.format(sparse_format, attr)).read())
	return attrs


def sparse_csx_matrix_to_hdf(obj, path, name):
	with tables.open_file(os.path.join(path, name), 'a') as f:
		for attr in ['data', 'indices', 'indptr', 'shape']:
			arr = np.asarray(getattr(obj, attr))
			atom = tables.Atom.from_dtype(arr.dtype)
			d = f.create_carray(f.root, 'csx_{}'.format(attr), atom, arr.shape)
			d[:] = arr


def sparse_coo_matrix_to_hdf(obj, path):

	# Data
	with tables.open_file(os.path.join(path, 'coo_matrix.hdf'), 'a') as f:
		atom = tables.Atom.from_dtype(obj.data.dtype)
		d = f.create_carray(f.root, 'coo_data', atom, obj.data.shape)
		d[:] = obj.data

	# Rows
	with tables.open_file(os.path.join(path, 'coo_matrix.hdf'), 'a') as f:
		atom = tables.Atom.from_dtype(obj.nonzero()[0].dtype)
		d = f.create_carray(f.root, 'coo_rows', atom, obj.nonzero()[0].shape)
		d[:] = obj.nonzero()[0]

	# Columns
	with tables.open_file(os.path.join(path, 'coo_matrix.hdf'), 'a') as f:
		atom = tables.Atom.from_dtype(obj.nonzero()[1].dtype)
		d = f.create_carray(f.root, 'coo_cols', atom, obj.nonzero()[1].shape)
		d[:] = obj.nonzero()[1]

	# Shape
	with tables.open_file(os.path.join(path, 'coo_matrix.hdf'), 'a') as f:
		atom = tables.Atom.from_dtype(np.asarray(obj.shape).dtype)
		d = f.create_carray(f.root, 'coo_shape', atom, np.asarray(obj.shape).shape)
		d[:] = np.asarray(obj.shape)