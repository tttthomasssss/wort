__author__ = 'thk22'
# PyTables code more or less shamelessly stolen from: http://stackoverflow.com/questions/11129429/storing-numpy-sparse-matrix-in-hdf5-pytables/11130235#11130235
import logging
import math
import os
import re
import sys
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


def determine_chunk_size(dtype_size, p=0.2):
	if ('linux' in sys.platform.lower()): # Linux
		try:
			o = os.popen('free -m').read().split('\n')
			header = re.split(r'\s+', o[0].lower().strip())
			idx = header.index('free')

			mem = re.split(r'\s+', o[1].lower().strip())
			available_mem = int(mem[idx])
			going_to_use_mem = available_mem * p
			chunk_size = math.floor(going_to_use_mem / dtype_size)

			logging.info('OS={}; chunk size={}'.format(sys.platform, chunk_size))
		except Exception as ex:
			logging.error('Failed to determine memory of system: {}; falling back on hardcoded value.'.format(ex))
			chunk_size = 100000000
		except: # Pokemon catching, if shit goes wrong, just fall back on a hardcoded value to not break everything
			logging.error('Failed to determine memory of system; falling back on hardcoded value.')
			chunk_size = 100000000
	elif ('darwin' in sys.platform.lower()): # OS X / BSD(?)
		try:
			available_mem = int(os.popen('sysctl -n hw.memsize').read().strip())
			going_to_use_mem = available_mem * p
			chunk_size = math.floor(going_to_use_mem / dtype_size)

			logging.info('OS={}; chunk size={}'.format(sys.platform, chunk_size))
		except Exception as ex:
			logging.error('Failed to determine memory of system: {}; falling back on hardcoded value.'.format(ex))
			chunk_size = 100000000
		except: # Pokemon catching, if shit goes wrong, just fall back on a hardcoded value to not break everything
			logging.error('Failed to determine memory of system; falling back on hardcoded value.')
			chunk_size = 100000000
	else: # Windows and other stuff, use hardcoded number
		logging.info('OS={}; mem allocation heuristic not implemented for OS. using hardcoded chunk size={}'.format(chunk_size))
		chunk_size = 100000000

	return chunk_size
