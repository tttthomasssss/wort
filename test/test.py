__author__ = 'thomas'
import os

from scipy.spatial import distance
import joblib
import numpy as np

from common import paths
from wort.vsm import VSMVectorizer
from wort.corpus_readers import FrostReader
from wort.corpus_readers import MovieReviewReader
from wort.corpus_readers import CSVStreamReader


def test_discoutils_loader():
	#from apt_toolkit.utils import vector_utils
	from discoutils.thesaurus_loader import Vectors

	#vecs = vector_utils.load_vector_cache('/Users/thomas/DevSandbox/EpicDataShelf/tag-lab/mitchell_lapata_2010/cached_filtered_vectors/wikipedia_lc_1_lemma-True_pos-False_vectors_filtered_min_count-50_min_features-50_cache.joblib')
	#disco_vectors = Vectors.from_dict_of_dicts(d=vecs)
	in_path = os.path.join(paths.get_dataset_path(), 'movie_reviews', 'wort_vectors')
	#in_path = os.path.join(paths.get_dataset_path(), 'frost')
	print('Loading Model from {}'.format(os.path.join(in_path, 'VSMVectorizer.joblib')))
	vec = joblib.load(os.path.join(in_path, 'VSMVectorizer.joblib'))
	print(vec.T_.shape)

	disco_vectors = Vectors.from_wort_model(vec)
	disco_vectors.init_sims(n_neighbors=10, knn='brute', nn_metric='cosine')

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
	vec = VSMVectorizer(window_size=3, min_frequency=2, cache_intermediary_results=True,
						cache_path=os.path.split(base_path)[0])

	vec.fit(f)
	joblib.dump(vec, os.path.join(os.path.split(base_path)[0], 'VSMVectorizer.joblib'), compress=3)


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


if (__name__ == '__main__'):
	#vectorize_kafka()
	#test_wikipedia()
	test_movie_reviews()
	#test_movie_reviews_from_cache()
	#test_frost()
	test_discoutils_loader()