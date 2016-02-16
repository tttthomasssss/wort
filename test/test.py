__author__ = 'thomas'
import os

from nltk.corpus import stopwords
from scipy import sparse
from scipy.spatial import distance
import joblib
import numpy as np

from common import paths
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from wort import utils
from wort.vsm import VSMVectorizer
from wort.corpus_readers import FrostReader
from wort.corpus_readers import MovieReviewReader
from wort.corpus_readers import CSVStreamReader
from wort.datasets import fetch_miller_charles_30_dataset
from wort.datasets import fetch_rubinstein_goodenough_65_dataset


def test_hdf():
	X = np.maximum(0, np.random.rand(5, 5) - 0.5)

	X_csr = sparse.csr_matrix(X)
	X_coo = sparse.coo_matrix(X)

	# To hdf
	print('to hdf...')
	#utils.sparse_matrix_to_hdf(X_coo, os.path.join(paths.get_dataset_path(), '_temp'))
	#utils.sparse_matrix_to_hdf(X_csr, os.path.join(paths.get_dataset_path(), '_temp'))
	utils.numpy_to_hdf(X, os.path.join(paths.get_dataset_path(), '_temp'), 'X')

	# From hdf
	print('from hdf...')
	#XX_csr = utils.hdf_to_sparse_matrix(os.path.join(paths.get_dataset_path(), '_temp'), 'csr')
	#XX_coo = utils.hdf_to_sparse_matrix(os.path.join(paths.get_dataset_path(), '_temp'), 'coo')
	XX = utils.hdf_to_numpy(os.path.join(paths.get_dataset_path(), '_temp'), 'X')

	#print('CSR check={}'.format(np.all(XX_csr==X_csr)))
	#print('COO check={}'.format(np.all(XX_coo==X_coo)))
	print('NUMPY check={}'.format(np.all(X==XX)))


def test_discoutils_loader():
	#from apt_toolkit.utils import vector_utils
	from discoutils.thesaurus_loader import Vectors

	#vecs = vector_utils.load_vector_cache('/Users/thomas/DevSandbox/EpicDataShelf/tag-lab/mitchell_lapata_2010/cached_filtered_vectors/wikipedia_lc_1_lemma-True_pos-False_vectors_filtered_min_count-50_min_features-50_cache.joblib')
	#disco_vectors = Vectors.from_dict_of_dicts(d=vecs)
	#in_path = os.path.join(paths.get_dataset_path(), 'movie_reviews', 'wort_vectors')
	#in_path = os.path.join(paths.get_dataset_path(), 'frost')
	in_path = os.path.join(paths.get_dataset_path(), 'wikipedia/wort_models')
	print('Loading Model from {}'.format(os.path.join(in_path, '')))
	vec = VSMVectorizer.load_from_file(in_path)
	print('Converting to DisCo representation...')

	disco_vectors = Vectors.from_wort_model(vec)
	print('Disco model done!')
	disco_vectors.init_sims(n_neighbors=20, knn='brute', nn_metric='cosine')
	print('init sims done!')

	print('good: {}'.format(disco_vectors.get_nearest_neighbours('good')))
	print('bad: {}'.format(disco_vectors.get_nearest_neighbours('bad')))
	print('terrible: {}'.format(disco_vectors.get_nearest_neighbours('terrible')))
	print('boring: {}'.format(disco_vectors.get_nearest_neighbours('boring')))
	print('exciting: {}'.format(disco_vectors.get_nearest_neighbours('exciting')))
	print('movie: {}'.format(disco_vectors.get_nearest_neighbours('movie')))
	print('film: {}'.format(disco_vectors.get_nearest_neighbours('film')))
	print('book: {}'.format(disco_vectors.get_nearest_neighbours('book')))
	print('watch: {}'.format(disco_vectors.get_nearest_neighbours('watch')))
	print('hero: {}'.format(disco_vectors.get_nearest_neighbours('hero')))
	print('heroine: {}'.format(disco_vectors.get_nearest_neighbours('heroine')))
	print('music: {}'.format(disco_vectors.get_nearest_neighbours('music')))
	print('plot: {}'.format(disco_vectors.get_nearest_neighbours('plot')))
	print('cinematography: {}'.format(disco_vectors.get_nearest_neighbours('cinematography')))
	print('actor: {}'.format(disco_vectors.get_nearest_neighbours('actor')))
	print('star: {}'.format(disco_vectors.get_nearest_neighbours('star')))
	print('blind: {}'.format(disco_vectors.get_nearest_neighbours('blind')))
	print('friend: {}'.format(disco_vectors.get_nearest_neighbours('friend')))
	print('companion: {}'.format(disco_vectors.get_nearest_neighbours('companion')))
	print('cat: {}'.format(disco_vectors.get_nearest_neighbours('cat')))
	print('dog: {}'.format(disco_vectors.get_nearest_neighbours('dog')))


def test_frost():
	base_path = os.path.join(paths.get_dataset_path(), 'frost', 'stopping_by_woods_on_a_snowy_evening.txt')
	f = FrostReader(base_path)
	vec = VSMVectorizer(window_size=3, min_frequency=1, context_window_weighting='aggressive')

	vec.fit(f)
	joblib.dump(vec, os.path.join(os.path.split(base_path)[0], 'VSMVectorizer.joblib'), compress=3)


def transform_wikipedia_from_cache():
	import logging
	logging.basicConfig(format='%(asctime)s: %(levelname)s - %(message)s', datefmt='[%d/%m/%Y %H:%M:%S %p]', level=logging.INFO)
	base_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors_min_freq_100')
	if (not os.path.exists(base_path)):
		os.makedirs(base_path)

	vec = VSMVectorizer(window_size=5, min_frequency=100, cache_intermediary_results=True, cache_path=base_path)

	vec = vec.weight_transformation_from_cache()
	logging.info('Transformed cache...')

	transformed_out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors_min_freq_100', 'transformed_vectors_min_freq_100')
	if (not os.path.exists(transformed_out_path)):
		os.makedirs(transformed_out_path)

	logging.info('storing...')
	if (sparse.issparse(vec.T_)):
		utils.sparse_matrix_to_hdf(vec.T_, transformed_out_path)
	else:
		utils.numpy_to_hdf(vec.T_, transformed_out_path, 'T')
	logging.info('stored')


def test_movie_reviews_from_cache():
	base_path = os.path.join(paths.get_dataset_path(), 'movie_reviews', 'wort_vectors')
	vec = VSMVectorizer(window_size=5, min_frequency=50, cache_intermediary_results=True, cache_path=base_path,
						sppmi_shift=5, weighting='sppmi')
	vec.weight_transformation_from_cache()

	print(vec.M_.shape)
	print(vec.M_.max())
	print(vec.M_.min())
	print('------------------')
	print(vec.T_.shape)
	print(vec.T_.max())
	print(vec.T_.min())


def test_movie_reviews():
	p = os.path.join(os.path.join(paths.get_dataset_path(), 'movie_reviews', 'aclImdb', 'unlabelled_docs'))
	mr = MovieReviewReader(p)

	out_path = os.path.join(paths.get_dataset_path(), 'movie_reviews', 'wort_vectors')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	vec = VSMVectorizer(window_size=10, min_frequency=50, cache_intermediary_results=True, cache_path=out_path)
	vec = vec.fit(mr)

	# Store model
	print('Dumping model to {}'.format(os.path.join(out_path, 'VSMVectorizer.joblib')))
	joblib.dump(vec, os.path.join(out_path, 'VSMVectorizer.joblib'), compress=3)
	print('SHAPE: ', vec.M_.shape)
	print('MAX: ', np.amax(vec.M_.A))
	print('MIN: ', np.amin(vec.M_.A))
	print('NNZ: {}; thats a ratio of {}'.format(np.count_nonzero(vec.M_.A), np.count_nonzero(vec.M_.A) / (vec.M_.shape[0] * vec.M_.shape[1])))


def test_wikipedia():
	p = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews.csv')
	wiki = CSVStreamReader(p)

	out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	vec = VSMVectorizer(window_size=5, min_frequency=50, cache_intermediary_results=True, cache_path=out_path)
	print(vec)

	# New fit
	#M = vec.fit(wiki)

	# From cache
	M = vec.weight_transformation_from_cache()

	joblib.dump(M, os.path.join(out_path, 'wikipedia_test'))


def vectorize_wikipedia():
	from discoutils.thesaurus_loader import Vectors
	from wort.datasets import get_miller_charles_30_words
	from wort.datasets import get_rubinstein_goodenough_65_words

	p = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews_lc_noid.tsv')
	wiki_reader = CSVStreamReader(p, delimiter='\t')

	out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors_min_freq_100')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	whitelist = list(get_miller_charles_30_words() | get_rubinstein_goodenough_65_words())

	for pmi_type in ['sppmi', 'ppmi']:
		for window_size in [2, 5]:
			print('CONFIG: pmi_type={}; window_size={}...'.format(pmi_type, window_size))
			vec = VSMVectorizer(window_size=window_size, min_frequency=100, cds=0.75, weighting=pmi_type, sppmi_shift=5,
								word_white_list=whitelist)

			vec.fit(wiki_reader)

			transformed_out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_model_pmi-{}_window-{}'.format(
				pmi_type, window_size
			))
			if (not os.path.exists(transformed_out_path)):
				os.makedirs(transformed_out_path)

			try:
				print('Saving to file')
				vec.save_to_file(transformed_out_path)
				print('Doing the DisCo business...')
			except OSError as ex:
				print('FAILFAILFAIL: {}'.format(ex))


def vectorize_kafka():

	# TODO: Check if PMI calculation is correct, compare to: https://github.com/mbatchkarov/DiscoUtils/blob/master/discoutils/reweighting.py

	docs = [
		'i sat on a table',
		'the cat sat on the mat.',
		'the pizza sat next to the table',
		'a green curry sat under the chair'
	]

	#vec = VSMVectorizer(window_size=2, min_frequency=2)
	#M_ppmi = vec.fit_transform(docs)

	with open(os.path.join(paths.get_dataset_path(), 'kafka', 'kafka_one_line_lc.txt'), mode='r', encoding='utf-8') as f:
		#vec = VSMVectorizer(window_size=5, cds=0.75, svd=300, svd_eig_weighting=0.5, sppmi_shift=5)
		vec = VSMVectorizer(window_size=5)
		M_ppmi = vec.fit_transform([f.read()])

		print ('PPMI Matrix created!')

	words = filter(lambda w: True if w in vec.inverted_index_.keys() else False, ['manager', 'director', 'clerk', 'innocent', 'judge', 'court', 'lawyer', 'law', 'josef', 'gregor', 'animal', 'samsa', 'trial', 'sister', 'father', 'mother', 'office', 'coat', 'work', 'fear', 'love', 'hate', 'manner', 'money', 'suit', 'custom', 'house', 'visitor'])

	for w in words:
		idx = vec.inverted_index_[w]

		min_dist = np.inf
		min_idx = -1

		for i in range(M_ppmi.shape[0]):
			if (i != idx):
				curr_dist = distance.cosine(M_ppmi[idx].A, M_ppmi[i].A)

				if (curr_dist < min_dist):
					min_idx = i
					min_dist = curr_dist

		print('\t[SIM=%.4f] WORD=%s; MOST SIMILAR=%s' % (min_dist, w, vec.index_[min_idx]))


def test_rg65_loader():
	ds = fetch_rubinstein_goodenough_65_dataset()

	print(ds)


def test_mc30_evaluation():
	ds = fetch_miller_charles_30_dataset()

	print('Loading Wort Model...')
	p = '/Users/thomas/DevSandbox/EpicDataShelf/tag-lab/wikipedia/wort_models/wort_models_cds-0.75_pmi-ppmi_ws-5_wctx-harmonic'
	wort_model = VSMVectorizer.load_from_file(path=p)
	print('Wort model loaded!')

	scores = []
	human_sims = []
	for w1, w2, sim in ds:
		if (w1 not in wort_model or w2 not in wort_model):
			print('\t[FAIL] - {} or {} not in model vocab!'.format(w1, w2))
		else:
			human_sims.append(sim)
			scores.append(1 - cosine(wort_model[w1].A, wort_model[w2].A))

	print('Spearman Rho: {}'.format(spearmanr(np.array(human_sims), np.array(scores))))


def test_rg65_evaluation():
	ds = fetch_rubinstein_goodenough_65_dataset()

	print('Loading Wort Model...')
	p = '/Users/thomas/DevSandbox/EpicDataShelf/tag-lab/wikipedia/wort_models/wort_models_cds-0.75_pmi-ppmi_ws-5_wctx-harmonic'
	wort_model = VSMVectorizer.load_from_file(path=p)
	print('Wort model loaded!')

	scores = []
	human_sims = []
	for w1, w2, sim in ds:
		if (w1 not in wort_model or w2 not in wort_model):
			print('\t[FAIL] - {} or {} not in model vocab!'.format(w1, w2))
		else:
			human_sims.append(sim)
			scores.append(1 - cosine(wort_model[w1].A, wort_model[w2].A))

	print('Spearman Rho: {}'.format(spearmanr(np.array(human_sims), np.array(scores))))

if (__name__ == '__main__'):
	#transform_wikipedia_from_cache()
	#vectorize_wikipedia()
	#vectorize_kafka()
	#test_wikipedia()
	#test_movie_reviews()
	#test_movie_reviews_from_cache()
	#test_frost()
	#test_discoutils_loader()
	#test_hdf()
	#test_rg65_loader()
	#test_rg65_evaluation()
	test_mc30_evaluation()