__author__ = 'thomas'
from argparse import ArgumentParser
import json
import logging
import math
import os

from nltk.corpus import stopwords
from scipy import sparse
from scipy.spatial import distance
import joblib
import numpy as np

try:
	from common import paths
except ImportError as ex:
	print(ex)
	print('Continuing anyway...')
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from wort.core import utils
from wort.core.utils import LemmaTokenizer
from wort.vsm import VSMVectorizer
from wort.corpus_readers import CSVStreamReader
from wort.corpus_readers import GzipStreamReader
from wort.corpus_readers import TextStreamReader
from wort.datasets import fetch_men_dataset
from wort.datasets import fetch_miller_charles_30_dataset
from wort.datasets import fetch_mturk_dataset
from wort.datasets import fetch_rare_words_dataset
from wort.datasets import fetch_rubinstein_goodenough_65_dataset
from wort.datasets import fetch_simlex_999_dataset
from wort.datasets import fetch_ws353_dataset
from wort.datasets import fetch_msr_syntactic_analogies_dataset
from wort.datasets import fetch_google_analogies_dataset
from wort.datasets import get_men_words
from wort.datasets import get_mturk_words
from wort.datasets import get_rare_words
from wort.datasets import get_ws353_words
from wort.datasets import get_simlex_999_words
from wort.datasets import get_msr_syntactic_analogies_words
from wort.datasets import get_google_analogies_words
from wort.datasets import get_rubinstein_goodenough_65_words
from wort.datasets import get_miller_charles_30_words
from wort.datasets import get_bless_words
from wort.evaluation import intrinsic_word_analogy_evaluation

parser = ArgumentParser()
parser.add_argument('-ip', '--input-path', type=str, help='path to input file')
parser.add_argument('-i', '--input-file', type=str, help='input file')
parser.add_argument('-op', '--output-path', type=str, help='path to output file')
parser.add_argument('-s', '--sample-size', type=int, help='sample size')
parser.add_argument('-cs', '--current-sample', type=int, help='current sample')
parser.add_argument('-cp', '--cache-path', type=str, help='path to cache')


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


def test_conll_reader():
	from wort.corpus_readers import CoNLLStreamReader
	r = CoNLLStreamReader(path='/Users/thomas/DevSandbox/EpicDataShelf/tag-lab/parsed_test/1.txt.tagged', data_index=1,
						  order='dep')

	vec = VSMVectorizer(window_size=2, min_frequency=2, weighting='ppmi', token_pattern=r'(?u)\b\w+\b')
	vec.fit(r)

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


def test_pizza():
	import math
	base_path = os.path.join(paths.get_dataset_path(), 'pizza_small', 'pizza_small.txt')
	f = CSVStreamReader(base_path)
	vec = VSMVectorizer(window_size=2, min_frequency=2, weighting='ppmi', token_pattern=r'(?u)\b\w+\b')

	vec.fit(f)

	vec.init_neighbours(num_neighbours=2)
	print('Pizza Neighbours: {}'.format(vec.neighbours('pizza')))
	print('Favourite Neighbours: {}'.format(vec.neighbours('favourite')))

	x = vec.transform('I love pizza and beer', composition='add', as_matrix=False)
	print(x)

	#joblib.dump(vec, os.path.join(os.path.split(base_path)[0], 'VSMVectorizer.joblib'), compress=3)


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


def vectorize_ukwac():
	#ukwac_reader = GzipStreamReader(path='/research/calps/data2/public/corpora/ukwac1.0/raw/ukwac_preproc.gz')
	#ukwac_reader = TextStreamReader(path='/lustre/scratch/inf/thk22/_datasets/ukwac/ukwac_lemmatised.txt')
	ukwac_reader = TextStreamReader(path='/media/data4/_datasets/ukwac_wackypedia_bnc/corpus/ukwac_wackypedia_bnc_lc_lemma.txt')

	out_path = os.path.join('/media/data4/_datasets/ukwac_wackypedia_bnc/', 'wort')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	if (not os.path.exists(os.path.join(out_path, 'ukwac_cooccurrence_cache'))):
		os.makedirs(os.path.join(out_path, 'ukwac_cooccurrence_cache'))

	#whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_mturk_words() | get_men_words() | get_rare_words() | get_simlex_999_words() | get_msr_syntactic_analogies_words() | get_google_analogies_words()
	#whitelist = get_ws353_words() | get_ws353_words(subset='similarity') | get_ws353_words(subset='relatedness')

	#print('Word whitelist contains {} words!'.format(len(whitelist)))
	import math
	for log_sppmi, sppmi in zip([0, math.log(5), math.log(10)], [0, 5, 10]):
		for pmi_type in ['ppmi']:
			for cds in [1., 0.75]:
				for window_size in [5, 2]:
					print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}...'.format(pmi_type, window_size, cds, sppmi))
					transformed_out_path = os.path.join('/media/data4/_datasets/ukwac_wackypedia_bnc/', 'wort', 'wort_model_ppmi_lemma-False_window-{}_cds-{}-sppmi_shift-{}'.format(
						window_size, cds, sppmi
					))
					if (not os.path.exists(transformed_out_path)):
						cache_path = os.path.join('/media/data4/_datasets/ukwac_wackypedia_bnc/', 'wort', 'wort_cache')
						if (not os.path.exists(cache_path)):
							os.makedirs(cache_path)

						vec = VSMVectorizer(window_size=window_size, min_frequency=50, cds=cds, weighting=pmi_type,
											sppmi_shift=log_sppmi, cache_path=cache_path, cache_intermediary_results=True)

						vec.fit(ukwac_reader)

						if (not os.path.exists(transformed_out_path)):
							os.makedirs(transformed_out_path)

						try:
							print('Saving to file')
							vec.save_to_file(transformed_out_path)
							print('Doing the DisCo business...')
						except OSError as ex:
							print('FAILFAILFAIL: {}'.format(ex))
					else:
						print('{} already exists!'.format(transformed_out_path))


def lemmatise_wikipedia():
	p = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews_lc_noid.tsv')
	wiki_reader = CSVStreamReader(p, delimiter='\t')
	ltk = LemmaTokenizer()

	with open(os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews_lc_noid_lemma.tsv'), 'w') as out_file:
		for idx, line in enumerate(wiki_reader, 1):
			new_line = ' '.join(ltk(line.strip()))
			out_file.write(new_line + '\n')
			if (idx % 10000 == 0): print('{} lines processed!'.format(idx))


def lemmatise_ukwac():
	ukwac_reader = GzipStreamReader(path='/research/calps/data2/public/corpora/ukwac1.0/raw/ukwac_preproc.gz')
	ltk = LemmaTokenizer()

	with open(os.path.join(paths.get_dataset_path(), 'ukwac', 'ukwac_lemmatised.txt'), 'w') as out_file:
		for idx, line in enumerate(ukwac_reader, 1):
			new_line = ' '.join(ltk(line.strip()))
			out_file.write(new_line + '\n')
			if (idx % 10000 == 0): print('{} lines processed!'.format(idx))


def lemmatise_bnc():
	reader = TextStreamReader(path='/lustre/scratch/inf/thk22/_datasets/bnc/corpus/bnc_lc.txt')
	ltk = LemmaTokenizer()

	with open(os.path.join(paths.get_dataset_path(), 'bnc', 'corpus', 'bnc_lc_lemma.txt'), 'w') as out_file:
		for idx, line in enumerate(reader, 1):
			new_line = ' '.join(ltk(line.strip()))
			out_file.write(new_line + '\n')
			if (idx % 10000 == 0): logging.info('{} lines processed!'.format(idx))


def lemmatise_wackypedia():
	reader = TextStreamReader(path='/media/data4/_datasets/wackypedia/corpus/wackypedia.txt')
	ltk = LemmaTokenizer()

	with open('/media/data4/_datasets/wackypedia/corpus/wackypedia_lc_lemma.txt', 'w') as out_file:
		for idx, line in enumerate(reader, 1):
			new_line = ' '.join(ltk(line.lower().strip()))
			out_file.write(new_line + '\n')
			if (idx % 10000 == 0): logging.info('{} lines processed!'.format(idx))


def lemmatise_gutenberg():
	#reader = CSVStreamReader(path='/lustre/scratch/inf/thk22/_datasets/gutenberg/corpus/gutenberg_lowercase-True.tsv', delimiter='\t')
	reader = CSVStreamReader(path='/infinity/_datasets/gutenberg/corpus/gutenberg_lowercase-True.tsv',
							 delimiter='\t')
	ltk = LemmaTokenizer()

	with open(os.path.join('/infinity/_datasets/gutenberg/corpus/', 'gutenberg_lc_lemma.txt'), 'w') as out_file:
		for idx, line in enumerate(reader, 1):
			new_line = ' '.join(ltk(line.strip()))
			out_file.write(new_line + '\n')
			if (idx % 10000 == 0): logging.info('{} lines processed!'.format(idx))


def lemmatise_toronto():
	reader = TextStreamReader(path='/infinity/_datasets/toronto_books_corpus/corpus/books_combined.txt', lowercase=True)
	ltk = LemmaTokenizer()

	with open(os.path.join('/infinity/_datasets/toronto_books_corpus/corpus/', 'books_combined_lc_lemma.txt'), 'w') as out_file:
		for idx, line in enumerate(reader, 1):
			new_line = ' '.join(ltk(line.strip()))
			out_file.write(new_line + '\n')
			if (idx % 10000 == 0): logging.info('{} lines processed!'.format(idx))


def lemmatise_gigaword():
	reader = TextStreamReader(path='/infinity/_datasets/gigaword/cleaned_texts/gigaword.txt', lowercase=True)
	ltk = LemmaTokenizer()

	with open(os.path.join('/infinity/_datasets/gigaword/cleaned_texts/', 'gigaword_lc_lemma.txt'), 'w') as out_file:
		for idx, line in enumerate(reader, 1):
			new_line = ' '.join(ltk(line.strip()))
			out_file.write(new_line + '\n')
			if (idx % 10000 == 0): logging.info('{} lines processed!'.format(idx))


def vectorize_pizza_epic():
	base_path = os.path.join(paths.get_dataset_path(), 'pizza_small', 'pizza_small.txt')
	f = CSVStreamReader(base_path)

	out_path = os.path.join(paths.get_dataset_path(), 'pizza', 'wort_vectors_min_freq_100')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	import math
	for log_sppmi, sppmi in zip([0, math.log(5), math.log(10), math.log(40), math.log(100)], [0, 5, 10, 40, 100]):
		for pmi_type in ['ppmi']:
			for cds in [1., 0.75]:
				for window_size in [1, 2, 5, 10]:# [5, 2]:
					for dim in [0, 50, 100, 300, 600, 1000, 5000]:
						for add_ctx_vectors in [False, True]:
							print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}...'.format(pmi_type, window_size, cds, sppmi))
							transformed_out_path = os.path.join(paths.get_dataset_path(), 'pizza', 'epic_wort',
								'wort_model_ppmi_lemma-True_window-{}_cds-{}-sppmi_shift-{}_dim-{}_add_ctx-{}'.format(
								window_size, cds, sppmi, dim, add_ctx_vectors
							))
							if (not os.path.exists(transformed_out_path)):
								cache_path = os.path.join(paths.get_dataset_path(), 'pizza', 'epic_wort_cache')
								if (not os.path.exists(cache_path)):
									os.makedirs(cache_path)

								red = None if dim <= 0 else 'svd'
								vec = VSMVectorizer(window_size=window_size, min_frequency=2, cds=cds, weighting=pmi_type,
													sppmi_shift=log_sppmi, cache_path=cache_path, cache_intermediary_results=True,
													dim_reduction=red, svd_dim=dim, add_context_vectors=add_ctx_vectors)

								vec.fit(f)

								if (not os.path.exists(transformed_out_path)):
									os.makedirs(transformed_out_path)

								try:
									print('Saving to file')
									vec.save_to_file(transformed_out_path)
									print('Doing the DisCo business...')
								except OSError as ex:
									print('FAILFAILFAIL: {}'.format(ex))
							else:
								print('{} already exists!'.format(transformed_out_path))


def vectorize_wikipedia_epic():
	from discoutils.thesaurus_loader import Vectors
	from wort.datasets import get_miller_charles_30_words
	from wort.datasets import get_rubinstein_goodenough_65_words

	p = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews_lc_noid_lemma.tsv')
	wiki_reader = CSVStreamReader(p, delimiter='\t')

	out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors_min_freq_100')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	#whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_mturk_words() | get_men_words() | get_rare_words() | get_simlex_999_words() | get_msr_syntactic_analogies_words() | get_google_analogies_words()
	whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_men_words() | get_simlex_999_words()
	# ML 2010 words
	#whitelist = ['achieve', 'acquire', 'action', 'activity', 'address', 'age', 'agency', 'air', 'allowance', 'american', 'amount', 'area', 'arm', 'ask', 'assembly', 'assistant', 'attend', 'attention', 'authority', 'basic', 'battle', 'bedroom', 'begin', 'benefit', 'better', 'black', 'board', 'body', 'book', 'building', 'bus', 'business', 'buy', 'call', 'capital', 'care', 'career', 'case', 'cause', 'central', 'centre', 'certain', 'charge', 'child', 'circumstance', 'city', 'close', 'club', 'cold', 'collect', 'college', 'committee', 'community', 'company', 'computer', 'condition', 'conference', 'consider', 'contract', 'control', 'cost', 'council', 'country', 'county', 'course', 'credit', 'cross', 'cut', 'dark', 'datum', 'day', 'defence', 'demand', 'department', 'develop', 'development', 'different', 'difficulty', 'director', 'discuss', 'door', 'drink', 'earlier', 'early', 'economic', 'economy', 'education', 'effect', 'effective', 'efficient', 'elderly', 'emphasise', 'encourage', 'end', 'environment', 'european', 'evening', 'event', 'evidence', 'example', 'exercise', 'express', 'eye', 'face', 'family', 'federal', 'fight', 'follow', 'football', 'form', 'further', 'future', 'game', 'general', 'good', 'government', 'great', 'group', 'hair', 'hall', 'hand', 'head', 'health', 'hear', 'help', 'high', 'hold', 'home', 'hot', 'house', 'housing', 'importance', 'important', 'increase', 'industrial', 'industry', 'influence', 'information', 'injury', 'intelligence', 'interest', 'intervention', 'issue', 'job', 'join', 'kind', 'kitchen', 'knowledge', 'labour', 'lady', 'land', 'language', 'large', 'law', 'leader', 'league', 'leave', 'left', 'letter', 'level', 'life', 'lift', 'like', 'line', 'little', 'local', 'long', 'loss', 'low', 'major', 'majority', 'man', 'management', 'manager', 'market', 'marketing', 'match', 'matter', 'meet', 'meeting', 'member', 'message', 'method', 'minister', 'modern', 'name', 'national', 'need', 'new', 'news', 'northern', 'number', 'offer', 'office', 'officer', 'official', 'oil', 'old', 'older', 'opposition', 'part', 'particular', 'party', 'pass', 'pay', 'people', 'period', 'person', 'personnel', 'phone', 'place', 'plan', 'planning', 'play', 'point', 'policy', 'political', 'pose', 'position', 'pour', 'power', 'practical', 'present', 'previous', 'price', 'principle', 'problem', 'produce', 'programme', 'project', 'property', 'provide', 'public', 'quantity', 'question', 'railway', 'raise', 'rate', 'reach', 'read', 'receive', 'reduce', 'region', 'remember', 'require', 'requirement', 'research', 'result', 'right', 'road', 'role', 'room', 'rule', 'rural', 'satisfy', 'secretary', 'security', 'sell', 'send', 'service', 'set', 'share', 'short', 'shut', 'significant', 'similar', 'situation', 'skill', 'small', 'social', 'special', 'stage', 'start', 'state', 'station', 'stress', 'stretch', 'structure', 'study', 'suffer', 'support', 'system', 'tax', 'tea', 'technique', 'technology', 'telephone', 'television', 'test', 'time', 'town', 'training', 'treatment', 'tv', 'unit', 'use', 'various', 'vast', 'view', 'wage', 'war', 'water', 'wave', 'way', 'weather', 'whole', 'win', 'window', 'woman', 'word', 'work', 'worker', 'world', 'write']

	print('Word whitelist contains {} words!'.format(len(whitelist)))
	import math
	for log_sppmi, sppmi in zip([0, math.log(5), math.log(10), math.log(40), math.log(100)], [0, 5, 10, 40, 100]):
		for pmi_type in ['ppmi']:
			for cds in [1., 0.75]:
				for window_size in [1, 2, 5, 10]:# [5, 2]:
					for dim in [0, 50, 100, 300, 600, 1000]:
						for add_ctx_vectors in [False, True]:
							print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}...'.format(pmi_type, window_size, cds, sppmi))
							transformed_out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'epic_wort',
								'wort_model_ppmi_lemma-True_window-{}_cds-{}-sppmi_shift-{}_dim-{}_add_ctx-{}'.format(
								window_size, cds, sppmi, dim, add_ctx_vectors
							))
							if (not os.path.exists(transformed_out_path)):
								cache_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'epic_wort_cache')
								if (not os.path.exists(cache_path)):
									os.makedirs(cache_path)

								red = None if dim <= 0 else 'svd'
								vec = VSMVectorizer(window_size=window_size, min_frequency=50, cds=cds, weighting=pmi_type,
													word_white_list=whitelist, sppmi_shift=log_sppmi, cache_path=cache_path,
													cache_intermediary_results=True, dim_reduction=red, svd_dim=dim,
													add_context_vectors=add_ctx_vectors)

								vec.fit(wiki_reader)

								if (not os.path.exists(transformed_out_path)):
									os.makedirs(transformed_out_path)

								try:
									print('Saving to file')
									vec.save_to_file(transformed_out_path)
									print('Doing the DisCo business...')
								except OSError as ex:
									print('FAILFAILFAIL: {}'.format(ex))
							else:
								print('{} already exists!'.format(transformed_out_path))


def test_token_and_vocab_count():
	p = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews_lc_noid_lemma.tsv')
	wiki_reader = CSVStreamReader(p, delimiter='\t')

	out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors_min_freq_100')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	#whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_mturk_words() | get_men_words() | get_rare_words() | get_simlex_999_words() | get_msr_syntactic_analogies_words() | get_google_analogies_words()
	whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_men_words() | get_simlex_999_words()

	print('----- W I K I P E D I A -----')
	print('Word whitelist contains {} words!'.format(len(whitelist)))
	import math
	for log_sppmi, sppmi in zip([0, math.log(5), math.log(10), math.log(40), math.log(100)], [0, 5, 10, 40, 100]):
		for pmi_type in ['ppmi']:
			for cds in [1., 0.75]:
				for window_size in [5]:# [5, 2]:
					print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}...'.format(pmi_type, window_size, cds, sppmi))
					transformed_out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_ml2010', 'wort_model_ppmi_lemma-True_window-{}_cds-{}-sppmi_shift-{}'.format(
						window_size, cds, sppmi
					))
					if (not os.path.exists(transformed_out_path)):
						cache_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_cache_ml2010', 'window_size-{}'.format(window_size))
						if (not os.path.exists(cache_path)):
							os.makedirs(cache_path)

						vec = VSMVectorizer(window_size=window_size, min_frequency=50, cds=cds, weighting=pmi_type,
											word_white_list=whitelist, sppmi_shift=log_sppmi, cache_path=cache_path,
											cache_intermediary_results=True)

						vec.fit_vocabulary(wiki_reader)

	print('----- U K W A C -----')
	ukwac_reader = TextStreamReader(path='/lustre/scratch/inf/thk22/_datasets/ukwac/ukwac_lemmatised.txt')

	out_path = os.path.join(paths.get_dataset_path(), 'ukwac', 'wort')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	if (not os.path.exists(os.path.join(out_path, 'ukwac_cooccurrence_cache'))):
		os.makedirs(os.path.join(out_path, 'ukwac_cooccurrence_cache'))

	#whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_mturk_words() | get_men_words() | get_rare_words() | get_simlex_999_words() | get_msr_syntactic_analogies_words() | get_google_analogies_words()
	whitelist = get_ws353_words() | get_men_words() | get_simlex_999_words()

	print('Word whitelist contains {} words!'.format(len(whitelist)))
	import math
	for log_sppmi, sppmi in zip([0, math.log(5), math.log(10), math.log(40), math.log(100)], [0, 5, 10, 40, 100]):
		for pmi_type in ['ppmi']:
			for cds in [1., 0.75]:
				for window_size in [5, 2, 1, 10]:
					print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}...'.format(pmi_type, window_size, cds, sppmi))
					transformed_out_path = os.path.join(paths.get_dataset_path(), 'ukwac', 'wort_model_ppmi_lemma-True_window-{}_cds-{}-sppmi_shift-{}'.format(
						window_size, cds, sppmi
					))
					if (not os.path.exists(transformed_out_path)):
						cache_path = os.path.join(paths.get_dataset_path(), 'ukwac', 'wort_cache', 'window_size-{}'.format(window_size))
						if (not os.path.exists(cache_path)):
							os.makedirs(cache_path)

						vec = VSMVectorizer(window_size=window_size, min_frequency=100, cds=cds, weighting=pmi_type,
											word_white_list=whitelist, sppmi_shift=log_sppmi, cache_path=cache_path,
											cache_intermediary_results=True)

						vec.fit_vocabulary(ukwac_reader)


def vectorize_bnc():
	#p = os.path.join(paths.get_dataset_path(), 'bnc', 'corpus', 'bnc_lc_lemma.txt')
	#p = os.path.join('/data/thk22/_datasets', 'bnc', 'corpus', 'bnc_lc_lemma.txt')
	p = '/mnt/data3/thk22/_datasets/bnc/corpus/bnc_lc_lemma.txt'
	bnc_reader = TextStreamReader(p)

	#whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_mturk_words() | get_men_words() | get_rare_words() | get_simlex_999_words() | get_msr_syntactic_analogies_words() | get_google_analogies_words()
	#whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_men_words() | get_simlex_999_words()
	#whitelist = get_ws353_words() | get_ws353_words(similarity_type='similarity') | get_ws353_words(similarity_type='relatedness') | get_men_words() | get_simlex_999_words()
	# ML 2010 words
	#whitelist = ['achieve', 'acquire', 'action', 'activity', 'address', 'age', 'agency', 'air', 'allowance', 'american', 'amount', 'area', 'arm', 'ask', 'assembly', 'assistant', 'attend', 'attention', 'authority', 'basic', 'battle', 'bedroom', 'begin', 'benefit', 'better', 'black', 'board', 'body', 'book', 'building', 'bus', 'business', 'buy', 'call', 'capital', 'care', 'career', 'case', 'cause', 'central', 'centre', 'certain', 'charge', 'child', 'circumstance', 'city', 'close', 'club', 'cold', 'collect', 'college', 'committee', 'community', 'company', 'computer', 'condition', 'conference', 'consider', 'contract', 'control', 'cost', 'council', 'country', 'county', 'course', 'credit', 'cross', 'cut', 'dark', 'datum', 'day', 'defence', 'demand', 'department', 'develop', 'development', 'different', 'difficulty', 'director', 'discuss', 'door', 'drink', 'earlier', 'early', 'economic', 'economy', 'education', 'effect', 'effective', 'efficient', 'elderly', 'emphasise', 'encourage', 'end', 'environment', 'european', 'evening', 'event', 'evidence', 'example', 'exercise', 'express', 'eye', 'face', 'family', 'federal', 'fight', 'follow', 'football', 'form', 'further', 'future', 'game', 'general', 'good', 'government', 'great', 'group', 'hair', 'hall', 'hand', 'head', 'health', 'hear', 'help', 'high', 'hold', 'home', 'hot', 'house', 'housing', 'importance', 'important', 'increase', 'industrial', 'industry', 'influence', 'information', 'injury', 'intelligence', 'interest', 'intervention', 'issue', 'job', 'join', 'kind', 'kitchen', 'knowledge', 'labour', 'lady', 'land', 'language', 'large', 'law', 'leader', 'league', 'leave', 'left', 'letter', 'level', 'life', 'lift', 'like', 'line', 'little', 'local', 'long', 'loss', 'low', 'major', 'majority', 'man', 'management', 'manager', 'market', 'marketing', 'match', 'matter', 'meet', 'meeting', 'member', 'message', 'method', 'minister', 'modern', 'name', 'national', 'need', 'new', 'news', 'northern', 'number', 'offer', 'office', 'officer', 'official', 'oil', 'old', 'older', 'opposition', 'part', 'particular', 'party', 'pass', 'pay', 'people', 'period', 'person', 'personnel', 'phone', 'place', 'plan', 'planning', 'play', 'point', 'policy', 'political', 'pose', 'position', 'pour', 'power', 'practical', 'present', 'previous', 'price', 'principle', 'problem', 'produce', 'programme', 'project', 'property', 'provide', 'public', 'quantity', 'question', 'railway', 'raise', 'rate', 'reach', 'read', 'receive', 'reduce', 'region', 'remember', 'require', 'requirement', 'research', 'result', 'right', 'road', 'role', 'room', 'rule', 'rural', 'satisfy', 'secretary', 'security', 'sell', 'send', 'service', 'set', 'share', 'short', 'shut', 'significant', 'similar', 'situation', 'skill', 'small', 'social', 'special', 'stage', 'start', 'state', 'station', 'stress', 'stretch', 'structure', 'study', 'suffer', 'support', 'system', 'tax', 'tea', 'technique', 'technology', 'telephone', 'television', 'test', 'time', 'town', 'training', 'treatment', 'tv', 'unit', 'use', 'various', 'vast', 'view', 'wage', 'war', 'water', 'wave', 'way', 'weather', 'whole', 'win', 'window', 'woman', 'word', 'work', 'worker', 'world', 'write']

	whitelist = get_ws353_words() | \
				get_men_words() | \
				get_simlex_999_words() | \
				get_ws353_words(similarity_type='similarity') | \
				get_ws353_words(similarity_type='relatedness') | \
				set(['achieve', 'acquire', 'action', 'activity', 'address', 'age', 'agency', 'air', 'allowance', 'american', 'amount', 'area', 'arm', 'ask', 'assembly', 'assistant', 'attend', 'attention', 'authority', 'basic', 'battle', 'bedroom', 'begin', 'benefit', 'better', 'black', 'board', 'body', 'book', 'building', 'bus', 'business', 'buy', 'call', 'capital', 'care', 'career', 'case', 'cause', 'central', 'centre', 'certain', 'charge', 'child', 'circumstance', 'city', 'close', 'club', 'cold', 'collect', 'college', 'committee', 'community', 'company', 'computer', 'condition', 'conference', 'consider', 'contract', 'control', 'cost', 'council', 'country', 'county', 'course', 'credit', 'cross', 'cut', 'dark', 'datum', 'day', 'defence', 'demand', 'department', 'develop', 'development', 'different', 'difficulty', 'director', 'discuss', 'door', 'drink', 'earlier', 'early', 'economic', 'economy', 'education', 'effect', 'effective', 'efficient', 'elderly', 'emphasise', 'encourage', 'end', 'environment', 'european', 'evening', 'event', 'evidence', 'example', 'exercise', 'express', 'eye', 'face', 'family', 'federal', 'fight', 'follow', 'football', 'form', 'further', 'future', 'game', 'general', 'good', 'government', 'great', 'group', 'hair', 'hall', 'hand', 'head', 'health', 'hear', 'help', 'high', 'hold', 'home', 'hot', 'house', 'housing', 'importance', 'important', 'increase', 'industrial', 'industry', 'influence', 'information', 'injury', 'intelligence', 'interest', 'intervention', 'issue', 'job', 'join', 'kind', 'kitchen', 'knowledge', 'labour', 'lady', 'land', 'language', 'large', 'law', 'leader', 'league', 'leave', 'left', 'letter', 'level', 'life', 'lift', 'like', 'line', 'little', 'local', 'long', 'loss', 'low', 'major', 'majority', 'man', 'management', 'manager', 'market', 'marketing', 'match', 'matter', 'meet', 'meeting', 'member', 'message', 'method', 'minister', 'modern', 'name', 'national', 'need', 'new', 'news', 'northern', 'number', 'offer', 'office', 'officer', 'official', 'oil', 'old', 'older', 'opposition', 'part', 'particular', 'party', 'pass', 'pay', 'people', 'period', 'person', 'personnel', 'phone', 'place', 'plan', 'planning', 'play', 'point', 'policy', 'political', 'pose', 'position', 'pour', 'power', 'practical', 'present', 'previous', 'price', 'principle', 'problem', 'produce', 'programme', 'project', 'property', 'provide', 'public', 'quantity', 'question', 'railway', 'raise', 'rate', 'reach', 'read', 'receive', 'reduce', 'region', 'remember', 'require', 'requirement', 'research', 'result', 'right', 'road', 'role', 'room', 'rule', 'rural', 'satisfy', 'secretary', 'security', 'sell', 'send', 'service', 'set', 'share', 'short', 'shut', 'significant', 'similar', 'situation', 'skill', 'small', 'social', 'special', 'stage', 'start', 'state', 'station', 'stress', 'stretch', 'structure', 'study', 'suffer', 'support', 'system', 'tax', 'tea', 'technique', 'technology', 'telephone', 'television', 'test', 'time', 'town', 'training', 'treatment', 'tv', 'unit', 'use', 'various', 'vast', 'view', 'wage', 'war', 'water', 'wave', 'way', 'weather', 'whole', 'win', 'window', 'woman', 'word', 'work', 'worker', 'world', 'write']) | \
				set(['argument', 'ball', 'beam', 'body', 'boom', 'bow', 'burn', 'burst', 'butler', 'chatter', 'child', 'cigar', 'cigarette', 'click', 'company', 'concentration', 'conflict', 'courage', 'decline', 'determination', 'digress', 'discussion', 'erupt', 'export', 'eye', 'face', 'falter', 'fear', 'fire', 'flame', 'flare', 'flick', 'flicker', 'flinch', 'flood', 'fluctuate', 'fountain', 'gabble', 'girl', 'glow', 'government', 'gun', 'hand', 'head', 'heart', 'hope', 'industry', 'interest', 'island', 'kick', 'lessen', 'machine', 'man', 'mind', 'noise', 'opinion', 'optimism', 'prosper', 'pulse', 'rally', 'rebound', 'recoil', 'reel', 'ricochet', 'rifle', 'roam', 'row', 'sale', 'screen', 'share', 'shot', 'shoulder', 'shudder', 'sink', 'skin', 'slouch', 'slump', 'stagger', 'stoop', 'storm', 'stray', 'submit', 'subside', 'symptom', 'temper', 'thought', 'throb', 'thunder', 'tongue', 'tooth', 'value', 'vein', 'voice', 'waver', 'whirl'])


	print('Word whitelist contains {} words!'.format(len(whitelist)))
	import math
	for log_sppmi, sppmi in zip([0, math.log(5), math.log(10)], [0, 5, 10]):
		for pmi_type in ['ppmi']:
			for cds in [1., 0.75]:
				for window_size in [1, 2, 5, 10]:#[3, 5, 10, 7]:
					for weighting_fn in ['constant']:#['constant', 'aggressive', 'very_aggressive', 'harmonic', 'distance', 'inverse_harmonic', 'gaussian']:
						for dim, reduction in zip([0], [None]):#zip([0, 25, 50, 100, 300], [None, 'svd', 'svd', 'svd', 'svd']):
							print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}; context_weighting={}...'.format(pmi_type, window_size, cds, sppmi, weighting_fn))
							#transformed_out_path = os.path.join(paths.get_dataset_path(), 'bnc', 'coling_wort', 'wort_model_ppmi_lemma-True_window-{}_cds-{}-sppmi_shift-{}_{}'.format(
							#	window_size, cds, sppmi, weighting_fn
							#))
							transformed_out_path = os.path.join('/mnt/data3/thk22/_datasets/bnc/', 'wort_vectors', 'wort_model_ppmi_lemma-True_window-{}_cds-{}-dim-{}_sppmi_shift-{}'.format(
								window_size, cds, '{}-{}'.format(reduction, dim), sppmi
							))
							if (not os.path.exists(transformed_out_path)):
								#cache_path = os.path.join(paths.get_dataset_path(), 'bnc', 'wort_cache_coling')
								cache_path = os.path.join('/mnt/data3/thk22/_datasets/bnc/corpus', 'wort_cache')
								if (not os.path.exists(cache_path)):
									os.makedirs(cache_path)

								vec = VSMVectorizer(window_size=window_size, min_frequency=30, cds=cds, weighting=pmi_type,
													word_white_list=whitelist, sppmi_shift=log_sppmi, cache_path=cache_path,
													context_window_weighting=weighting_fn, cache_intermediary_results=True,
													dim_reduction=reduction, dim_reduction_kwargs={'dimensionality': dim})

								vec.fit(bnc_reader)

								if (not os.path.exists(transformed_out_path)):
									os.makedirs(transformed_out_path)

								try:
									print('Saving to file')
									vec.save_to_file(transformed_out_path)
									print('Doing the DisCo business...')
								except OSError as ex:
									print('FAILFAILFAIL: {}'.format(ex))
						else:
							print('{} already exists!'.format(transformed_out_path))


def vectorize_bnc_samples(input_file, output_path, cache_path, current_sample):
	bnc_reader = TextStreamReader(input_file)

	'''
	whitelist = get_ws353_words() | \
				get_men_words() | \
				get_simlex_999_words() | \
				get_ws353_words(similarity_type='similarity') | \
				get_ws353_words(similarity_type='relatedness') | \
				set(['achieve', 'acquire', 'action', 'activity', 'address', 'age', 'agency', 'air', 'allowance', 'american', 'amount', 'area', 'arm', 'ask', 'assembly', 'assistant', 'attend', 'attention', 'authority', 'basic', 'battle', 'bedroom', 'begin', 'benefit', 'better', 'black', 'board', 'body', 'book', 'building', 'bus', 'business', 'buy', 'call', 'capital', 'care', 'career', 'case', 'cause', 'central', 'centre', 'certain', 'charge', 'child', 'circumstance', 'city', 'close', 'club', 'cold', 'collect', 'college', 'committee', 'community', 'company', 'computer', 'condition', 'conference', 'consider', 'contract', 'control', 'cost', 'council', 'country', 'county', 'course', 'credit', 'cross', 'cut', 'dark', 'datum', 'day', 'defence', 'demand', 'department', 'develop', 'development', 'different', 'difficulty', 'director', 'discuss', 'door', 'drink', 'earlier', 'early', 'economic', 'economy', 'education', 'effect', 'effective', 'efficient', 'elderly', 'emphasise', 'encourage', 'end', 'environment', 'european', 'evening', 'event', 'evidence', 'example', 'exercise', 'express', 'eye', 'face', 'family', 'federal', 'fight', 'follow', 'football', 'form', 'further', 'future', 'game', 'general', 'good', 'government', 'great', 'group', 'hair', 'hall', 'hand', 'head', 'health', 'hear', 'help', 'high', 'hold', 'home', 'hot', 'house', 'housing', 'importance', 'important', 'increase', 'industrial', 'industry', 'influence', 'information', 'injury', 'intelligence', 'interest', 'intervention', 'issue', 'job', 'join', 'kind', 'kitchen', 'knowledge', 'labour', 'lady', 'land', 'language', 'large', 'law', 'leader', 'league', 'leave', 'left', 'letter', 'level', 'life', 'lift', 'like', 'line', 'little', 'local', 'long', 'loss', 'low', 'major', 'majority', 'man', 'management', 'manager', 'market', 'marketing', 'match', 'matter', 'meet', 'meeting', 'member', 'message', 'method', 'minister', 'modern', 'name', 'national', 'need', 'new', 'news', 'northern', 'number', 'offer', 'office', 'officer', 'official', 'oil', 'old', 'older', 'opposition', 'part', 'particular', 'party', 'pass', 'pay', 'people', 'period', 'person', 'personnel', 'phone', 'place', 'plan', 'planning', 'play', 'point', 'policy', 'political', 'pose', 'position', 'pour', 'power', 'practical', 'present', 'previous', 'price', 'principle', 'problem', 'produce', 'programme', 'project', 'property', 'provide', 'public', 'quantity', 'question', 'railway', 'raise', 'rate', 'reach', 'read', 'receive', 'reduce', 'region', 'remember', 'require', 'requirement', 'research', 'result', 'right', 'road', 'role', 'room', 'rule', 'rural', 'satisfy', 'secretary', 'security', 'sell', 'send', 'service', 'set', 'share', 'short', 'shut', 'significant', 'similar', 'situation', 'skill', 'small', 'social', 'special', 'stage', 'start', 'state', 'station', 'stress', 'stretch', 'structure', 'study', 'suffer', 'support', 'system', 'tax', 'tea', 'technique', 'technology', 'telephone', 'television', 'test', 'time', 'town', 'training', 'treatment', 'tv', 'unit', 'use', 'various', 'vast', 'view', 'wage', 'war', 'water', 'wave', 'way', 'weather', 'whole', 'win', 'window', 'woman', 'word', 'work', 'worker', 'world', 'write']) | \
				set(['argument', 'ball', 'beam', 'body', 'boom', 'bow', 'burn', 'burst', 'butler', 'chatter', 'child', 'cigar', 'cigarette', 'click', 'company', 'concentration', 'conflict', 'courage', 'decline', 'determination', 'digress', 'discussion', 'erupt', 'export', 'eye', 'face', 'falter', 'fear', 'fire', 'flame', 'flare', 'flick', 'flicker', 'flinch', 'flood', 'fluctuate', 'fountain', 'gabble', 'girl', 'glow', 'government', 'gun', 'hand', 'head', 'heart', 'hope', 'industry', 'interest', 'island', 'kick', 'lessen', 'machine', 'man', 'mind', 'noise', 'opinion', 'optimism', 'prosper', 'pulse', 'rally', 'rebound', 'recoil', 'reel', 'ricochet', 'rifle', 'roam', 'row', 'sale', 'screen', 'share', 'shot', 'shoulder', 'shudder', 'sink', 'skin', 'slouch', 'slump', 'stagger', 'stoop', 'storm', 'stray', 'submit', 'subside', 'symptom', 'temper', 'thought', 'throb', 'thunder', 'tongue', 'tooth', 'value', 'vein', 'voice', 'waver', 'whirl'])
	'''
	whitelist = get_bless_words()

	print('Word whitelist contains {} words!'.format(len(whitelist)))
	import math
	for log_sppmi, sppmi in zip([math.log(1), math.log(5), math.log(10), math.log(40), math.log(100)], [0, 5, 10, 40, 100]):#zip([0, math.log(5), math.log(10)], [0, 5, 10]):
		for pmi_type in ['ppmi']:
			for cds in [1., 0.75]:
				for window_size in [1, 2, 5, 10]:#[3, 5, 10, 7]:
					for weighting_fn in ['constant']:#['constant', 'aggressive', 'very_aggressive', 'harmonic', 'distance', 'inverse_harmonic', 'gaussian']:
						for dim, reduction in zip([0], [None]):#zip([0, 25, 50, 100, 300], [None, 'svd', 'svd', 'svd', 'svd']):
							print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}; context_weighting={}...'.format(pmi_type, window_size, cds, sppmi, weighting_fn))
							#transformed_out_path = os.path.join(paths.get_dataset_path(), 'bnc', 'coling_wort', 'wort_model_ppmi_lemma-True_window-{}_cds-{}-sppmi_shift-{}_{}'.format(
							#	window_size, cds, sppmi, weighting_fn
							#))
							transformed_out_path = os.path.join(output_path, 'wort_model_ppmi_lemma-True_window-{}_cds-{}-dim-{}_sppmi_shift-{}'.format(
								window_size, cds, '{}-{}'.format(reduction, dim), sppmi
							))
							if (not os.path.exists(transformed_out_path)):
								#cache_path = os.path.join(paths.get_dataset_path(), 'bnc', 'wort_cache_coling')
								if (not os.path.exists(cache_path)):
									os.makedirs(cache_path)

								vec = VSMVectorizer(window_size=window_size, min_frequency=30, cds=cds, weighting=pmi_type,
													word_white_list=whitelist, sppmi_shift=log_sppmi, cache_path=cache_path,
													context_window_weighting=weighting_fn, cache_intermediary_results=True,
													dim_reduction=reduction, dim_reduction_kwargs={'dimensionality': dim},
													decode_error='replace')

								vec.fit(bnc_reader)

								if (not os.path.exists(transformed_out_path)):
									os.makedirs(transformed_out_path)

								try:
									print('Saving to file')
									vec.save_to_file(transformed_out_path)
									print('Doing the DisCo business...')
								except OSError as ex:
									print('FAILFAILFAIL: {}'.format(ex))
						else:
							print('{} already exists!'.format(transformed_out_path))


def vectorize_amazon_reviews():
	from discoutils.thesaurus_loader import Vectors
	from wort.datasets import get_miller_charles_30_words
	from wort.datasets import get_rubinstein_goodenough_65_words

	# p = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews_lc_noid_lemma.tsv')
	p = '/infinity/_datasets/amazon_reviews/reviews_movies_and_tv_lc.txt'
	wiki_reader = TextStreamReader(p)

	# out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors')
	out_path = os.path.join('/bkp/thk22/_datasets/amazon_reviews/', 'wort_vectors')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	# whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_mturk_words() | get_men_words() | get_rare_words() | get_simlex_999_words() | get_msr_syntactic_analogies_words() | get_google_analogies_words()
	# whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_men_words() | get_simlex_999_words()
	whitelist = get_ws353_words() | get_ws353_words(similarity_type='similarity') | get_ws353_words(similarity_type='relatedness') | get_men_words() | get_simlex_999_words()
	# ML 2010 words
	# whitelist = ['achieve', 'acquire', 'action', 'activity', 'address', 'age', 'agency', 'air', 'allowance', 'american', 'amount', 'area', 'arm', 'ask', 'assembly', 'assistant', 'attend', 'attention', 'authority', 'basic', 'battle', 'bedroom', 'begin', 'benefit', 'better', 'black', 'board', 'body', 'book', 'building', 'bus', 'business', 'buy', 'call', 'capital', 'care', 'career', 'case', 'cause', 'central', 'centre', 'certain', 'charge', 'child', 'circumstance', 'city', 'close', 'club', 'cold', 'collect', 'college', 'committee', 'community', 'company', 'computer', 'condition', 'conference', 'consider', 'contract', 'control', 'cost', 'council', 'country', 'county', 'course', 'credit', 'cross', 'cut', 'dark', 'datum', 'day', 'defence', 'demand', 'department', 'develop', 'development', 'different', 'difficulty', 'director', 'discuss', 'door', 'drink', 'earlier', 'early', 'economic', 'economy', 'education', 'effect', 'effective', 'efficient', 'elderly', 'emphasise', 'encourage', 'end', 'environment', 'european', 'evening', 'event', 'evidence', 'example', 'exercise', 'express', 'eye', 'face', 'family', 'federal', 'fight', 'follow', 'football', 'form', 'further', 'future', 'game', 'general', 'good', 'government', 'great', 'group', 'hair', 'hall', 'hand', 'head', 'health', 'hear', 'help', 'high', 'hold', 'home', 'hot', 'house', 'housing', 'importance', 'important', 'increase', 'industrial', 'industry', 'influence', 'information', 'injury', 'intelligence', 'interest', 'intervention', 'issue', 'job', 'join', 'kind', 'kitchen', 'knowledge', 'labour', 'lady', 'land', 'language', 'large', 'law', 'leader', 'league', 'leave', 'left', 'letter', 'level', 'life', 'lift', 'like', 'line', 'little', 'local', 'long', 'loss', 'low', 'major', 'majority', 'man', 'management', 'manager', 'market', 'marketing', 'match', 'matter', 'meet', 'meeting', 'member', 'message', 'method', 'minister', 'modern', 'name', 'national', 'need', 'new', 'news', 'northern', 'number', 'offer', 'office', 'officer', 'official', 'oil', 'old', 'older', 'opposition', 'part', 'particular', 'party', 'pass', 'pay', 'people', 'period', 'person', 'personnel', 'phone', 'place', 'plan', 'planning', 'play', 'point', 'policy', 'political', 'pose', 'position', 'pour', 'power', 'practical', 'present', 'previous', 'price', 'principle', 'problem', 'produce', 'programme', 'project', 'property', 'provide', 'public', 'quantity', 'question', 'railway', 'raise', 'rate', 'reach', 'read', 'receive', 'reduce', 'region', 'remember', 'require', 'requirement', 'research', 'result', 'right', 'road', 'role', 'room', 'rule', 'rural', 'satisfy', 'secretary', 'security', 'sell', 'send', 'service', 'set', 'share', 'short', 'shut', 'significant', 'similar', 'situation', 'skill', 'small', 'social', 'special', 'stage', 'start', 'state', 'station', 'stress', 'stretch', 'structure', 'study', 'suffer', 'support', 'system', 'tax', 'tea', 'technique', 'technology', 'telephone', 'television', 'test', 'time', 'town', 'training', 'treatment', 'tv', 'unit', 'use', 'various', 'vast', 'view', 'wage', 'war', 'water', 'wave', 'way', 'weather', 'whole', 'win', 'window', 'woman', 'word', 'work', 'worker', 'world', 'write']

	print('Word whitelist contains {} words!'.format(len(whitelist)))
	import math
	for log_sppmi, sppmi in zip([0, math.log(5), math.log(10)], [0, 5, 10]):
		for pmi_type in ['ppmi']:
			for cds in [1., 0.75]:
				for window_size in [2, 1, 5]:  # [5, 2]:
					for dim in [50, 100, 300]:
						print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}; dim={}...'.format(pmi_type, window_size, cds, sppmi, dim))
						transformed_out_path = os.path.join('/bkp/thk22/_datasets/amazon_reviews/', 'wort_vectors', 'wort_model_ppmi_lemma-True_window-{}_cds-{}-sppmi_shift-{}_dim={}'.format(
							window_size, cds, sppmi, dim
						))
						if (not os.path.exists(transformed_out_path)):
							cache_path = os.path.join('/bkp/thk22/_datasets/amazon_reviews/', 'wort_cache')
							if (not os.path.exists(cache_path)):
								os.makedirs(cache_path)

							vec = VSMVectorizer(window_size=window_size, min_frequency=50, cds=cds, weighting=pmi_type,
												word_white_list=whitelist, sppmi_shift=log_sppmi, cache_path=cache_path,
												cache_intermediary_results=True, dim_reduction='svd', dim_reduction_kwargs={'dimensionality': dim})

							vec.fit(wiki_reader)

							if (not os.path.exists(transformed_out_path)):
								os.makedirs(transformed_out_path)

							try:
								print('Saving to file')
								vec.save_to_file(transformed_out_path)
								print('Doing the DisCo business...')
							except OSError as ex:
								print('FAILFAILFAIL: {}'.format(ex))
						else:
							print('{} already exists!'.format(transformed_out_path))


def vectorize_wikipedia():
	from discoutils.thesaurus_loader import Vectors
	from wort.datasets import get_miller_charles_30_words
	from wort.datasets import get_rubinstein_goodenough_65_words

	#p = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews_lc_noid_lemma.tsv')
	p = '/mnt/data0/thk22/_datasets/wikipedia/corpus/wikipedia_utf8_filtered_20pageviews_noid.csv'
	wiki_reader = CSVStreamReader(p, delimiter='\t')

	#out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors')
	out_path = os.path.join('/mnt/data0/thk22/_datasets/wikipedia/corpus/phd_thesis', 'wort_vectors')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	#whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_mturk_words() | get_men_words() | get_rare_words() | get_simlex_999_words() | get_msr_syntactic_analogies_words() | get_google_analogies_words()
	#whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_men_words() | get_simlex_999_words()
	#whitelist = get_ws353_words() | get_ws353_words(similarity_type='similarity') | get_ws353_words(similarity_type='relatedness') | get_men_words() | get_simlex_999_words()
	# Bless words
	whitelist = 'cloak,screwdriver,spade,corkscrew,car,bed,birch,squirrel,cockroach,bowl,apricot,clarinet,shovel,spinach,bomber,cow,beetle,glider,herring,acacia,pineapple,sofa,whale,cypress,knife,cedar,ant,jet,revolver,corn,deer,fridge,stereo,yacht,horse,wasp,vest,missile,tiger,cat,hornet,donkey,snake,turtle,sweater,cranberry,strawberry,elm,beet,gun,coconut,willow,grape,train,ferry,violin,pine,gorilla,lion,table,poplar,axe,bag,fox,tanker,chisel,hammer,cello,mug,lime,alligator,falcon,crow,dresser,dove,sword,oven,saw,rabbit,elephant,cucumber,carp,cod,dagger,spear,butterfly,robin,coyote,villa,bookcase,freezer,grasshopper,cabbage,scooter,helicopter,goat,flute,truck,lizard,penguin,library,washer,bear,tuna,robe,pigeon,bull,vulture,fighter,oak,castle,owl,sparrow,catfish,parsley,glove,hotel,pig,bottle,rifle,plum,coat,rake,wrench,turnip,television,scarf,grenade,goose,eagle,box,cathedral,cannon,lettuce,rat,couch,jar,toaster,blouse,hawk,broccoli,apple,carrot,frigate,peach,giraffe,celery,potato,pear,wardrobe,cherry,cauliflower,phone,stove,trumpet,hatchet,chair,fork,trout,battleship,desk,piano,woodpecker,saxophone,onion,spoon,bus,mackerel,goldfish,moth,pistol,pheasant,guitar,grapefruit,radish,radio,lemon,sieve,musket,ambulance,van,salmon,banana,garlic,beaver,restaurant,dress,shirt,dishwasher,dolphin,swan,cottage,hospital,pub,sheep,jacket,hat,bomb,frog,motorcycle'.split(',')
	# ML 2010 words
	#whitelist = ['achieve', 'acquire', 'action', 'activity', 'address', 'age', 'agency', 'air', 'allowance', 'american', 'amount', 'area', 'arm', 'ask', 'assembly', 'assistant', 'attend', 'attention', 'authority', 'basic', 'battle', 'bedroom', 'begin', 'benefit', 'better', 'black', 'board', 'body', 'book', 'building', 'bus', 'business', 'buy', 'call', 'capital', 'care', 'career', 'case', 'cause', 'central', 'centre', 'certain', 'charge', 'child', 'circumstance', 'city', 'close', 'club', 'cold', 'collect', 'college', 'committee', 'community', 'company', 'computer', 'condition', 'conference', 'consider', 'contract', 'control', 'cost', 'council', 'country', 'county', 'course', 'credit', 'cross', 'cut', 'dark', 'datum', 'day', 'defence', 'demand', 'department', 'develop', 'development', 'different', 'difficulty', 'director', 'discuss', 'door', 'drink', 'earlier', 'early', 'economic', 'economy', 'education', 'effect', 'effective', 'efficient', 'elderly', 'emphasise', 'encourage', 'end', 'environment', 'european', 'evening', 'event', 'evidence', 'example', 'exercise', 'express', 'eye', 'face', 'family', 'federal', 'fight', 'follow', 'football', 'form', 'further', 'future', 'game', 'general', 'good', 'government', 'great', 'group', 'hair', 'hall', 'hand', 'head', 'health', 'hear', 'help', 'high', 'hold', 'home', 'hot', 'house', 'housing', 'importance', 'important', 'increase', 'industrial', 'industry', 'influence', 'information', 'injury', 'intelligence', 'interest', 'intervention', 'issue', 'job', 'join', 'kind', 'kitchen', 'knowledge', 'labour', 'lady', 'land', 'language', 'large', 'law', 'leader', 'league', 'leave', 'left', 'letter', 'level', 'life', 'lift', 'like', 'line', 'little', 'local', 'long', 'loss', 'low', 'major', 'majority', 'man', 'management', 'manager', 'market', 'marketing', 'match', 'matter', 'meet', 'meeting', 'member', 'message', 'method', 'minister', 'modern', 'name', 'national', 'need', 'new', 'news', 'northern', 'number', 'offer', 'office', 'officer', 'official', 'oil', 'old', 'older', 'opposition', 'part', 'particular', 'party', 'pass', 'pay', 'people', 'period', 'person', 'personnel', 'phone', 'place', 'plan', 'planning', 'play', 'point', 'policy', 'political', 'pose', 'position', 'pour', 'power', 'practical', 'present', 'previous', 'price', 'principle', 'problem', 'produce', 'programme', 'project', 'property', 'provide', 'public', 'quantity', 'question', 'railway', 'raise', 'rate', 'reach', 'read', 'receive', 'reduce', 'region', 'remember', 'require', 'requirement', 'research', 'result', 'right', 'road', 'role', 'room', 'rule', 'rural', 'satisfy', 'secretary', 'security', 'sell', 'send', 'service', 'set', 'share', 'short', 'shut', 'significant', 'similar', 'situation', 'skill', 'small', 'social', 'special', 'stage', 'start', 'state', 'station', 'stress', 'stretch', 'structure', 'study', 'suffer', 'support', 'system', 'tax', 'tea', 'technique', 'technology', 'telephone', 'television', 'test', 'time', 'town', 'training', 'treatment', 'tv', 'unit', 'use', 'various', 'vast', 'view', 'wage', 'war', 'water', 'wave', 'way', 'weather', 'whole', 'win', 'window', 'woman', 'word', 'work', 'worker', 'world', 'write']
	
	print('Word whitelist contains {} words!'.format(len(whitelist)))
	import math
	for log_sppmi, sppmi in zip([0], [0]):#zip([0, math.log(5), math.log(10), math.log(20), math.log(40), math.log(100), math.log(250)], [0, 5, 10, 20, 40, 100, 250]):
		for pmi_type in ['ppmi']:
			for cds in [1.]:#[1., 0.75]:
				for window_size in [10, 20]:#[2, 1, 5]:# [5, 2]:
					#for dim in [50, 100, 300]:
					print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}; ...'.format(pmi_type, window_size, cds, sppmi))
					transformed_out_path = os.path.join('/mnt/data0/thk22/_datasets/wikipedia/corpus/phd_thesis', 'wort_vectors', 'wort_model_ppmi_lemma-False_pos-False_window-{}_cds-{}-sppmi_shift-{}'.format(
						window_size, cds, sppmi
					))
					if (not os.path.exists(transformed_out_path)):
						cache_path = os.path.join('/mnt/data0/thk22/_datasets/', 'wikipedia', 'wort_cache', 'phd_thesis')
						if (not os.path.exists(cache_path)):
							os.makedirs(cache_path)

						vec = VSMVectorizer(window_size=window_size, min_frequency=50, cds=cds, weighting=pmi_type,
											word_white_list=whitelist, sppmi_shift=log_sppmi, cache_path=cache_path,
											cache_intermediary_results=True)

						vec.fit(wiki_reader)

						if (not os.path.exists(transformed_out_path)):
							os.makedirs(transformed_out_path)

						try:
							print('Saving to file')
							vec.save_to_file(transformed_out_path)
							print('Doing the DisCo business...')
						except OSError as ex:
							print('FAILFAILFAIL: {}'.format(ex))
					else:
						print('{} already exists!'.format(transformed_out_path))


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
				curr_dist = distance.cosine(M_ppmi[idx], M_ppmi[i])

				if (curr_dist < min_dist):
					min_idx = i
					min_dist = curr_dist

		print('\t[SIM=%.4f] WORD=%s; MOST SIMILAR=%s' % (min_dist, w, vec.index_[min_idx]))


def test_rg65_loader():
	ds = fetch_rubinstein_goodenough_65_dataset()

	print(ds)


def test_mc30_evaluation(dataset='wikipedia'):
	print('MC30 Evaluation')
	ds = fetch_miller_charles_30_dataset()

	base_path = paths.get_dataset_path()

	scores_by_model = {}

	for wort_model_name in [
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-5',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-5'
	]:
		print('Loading Wort Model: {}...'.format(wort_model_name))
		wort_path = os.path.join(base_path, dataset, wort_model_name)
		wort_model = VSMVectorizer.load_from_file(path=wort_path)
		print('Wort model loaded!')

		scores = []
		human_sims = []
		for w1, w2, sim in ds:
			if (w1 not in wort_model or w2 not in wort_model):
				print('\t[FAIL] - {} or {} not in model vocab!'.format(w1, w2))
			else:
				human_sims.append(sim)
				sim = 1 - cosine(wort_model[w1].A, wort_model[w2].A)
				if (math.isnan(sim)):
					sim = 0
				scores.append(sim)

		spearman = spearmanr(np.array(human_sims), np.array(scores))
		scores_by_model[wort_model_name] = spearman
		print('[MC30] Spearman Rho: {}'.format(spearman))
		print('==================================================================================')

	return scores_by_model


def test_rg65_evaluation(dataset='wikipedia'):
	print('RG65 Evaluation')
	ds = fetch_rubinstein_goodenough_65_dataset()
	base_path = paths.get_dataset_path()

	scores_by_model = {}

	for wort_model_name in [
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-5',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-5'
	]:
		print('Loading Wort Model: {}...'.format(wort_model_name))
		wort_path = os.path.join(base_path, dataset, wort_model_name)
		wort_model = VSMVectorizer.load_from_file(path=wort_path)
		print('Wort model loaded!')

		scores = []
		human_sims = []
		for w1, w2, sim in ds:
			if (w1 not in wort_model or w2 not in wort_model):
				print('\t[FAIL] - {} or {} not in model vocab!'.format(w1, w2))
			else:
				human_sims.append(sim)
				sim = 1 - cosine(wort_model[w1].A, wort_model[w2].A)
				if (math.isnan(sim)):
					sim = 0
				scores.append(sim)

		spearman = spearmanr(np.array(human_sims), np.array(scores))
		scores_by_model[wort_model_name] = spearman
		print('[RG65] Spearman Rho: {}'.format(spearman))
		print('==================================================================================')

	return scores_by_model


def test_rw_evaluation(dataset='wikipedia'):
	print('RW Evaluation')
	ds = fetch_rare_words_dataset()
	base_path = paths.get_dataset_path()

	scores_by_model = {}

	for wort_model_name in [
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-5',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-5'
	]:
		print('Loading Wort Model: {}...'.format(wort_model_name))
		wort_path = os.path.join(base_path, dataset, wort_model_name)
		wort_model = VSMVectorizer.load_from_file(path=wort_path)
		print('Wort model loaded!')

		scores = []
		human_sims = []
		for w1, w2, sim in ds:
			if (w1 not in wort_model or w2 not in wort_model):
				print('\t[FAIL] - {} or {} not in model vocab!'.format(w1, w2))
			else:
				human_sims.append(sim)
				sim = 1 - cosine(wort_model[w1].A, wort_model[w2].A)
				if (math.isnan(sim)):
					sim = 0
				scores.append(sim)

		spearman = spearmanr(np.array(human_sims), np.array(scores))
		scores_by_model[wort_model_name] = spearman
		print('[RW] Spearman Rho: {}'.format(spearman))
		print('==================================================================================')

	return scores_by_model


def test_men_evaluation(dataset='wikipedia'):
	print('MEN Evaluation')
	ds = fetch_men_dataset()
	base_path = paths.get_dataset_path()

	scores_by_model = {}

	for wort_model_name in [
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-5',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-5'
	]:
		print('Loading Wort Model: {}...'.format(wort_model_name))
		wort_path = os.path.join(base_path, dataset, wort_model_name)
		wort_model = VSMVectorizer.load_from_file(path=wort_path)
		print('Wort model loaded!')

		scores = []
		human_sims = []
		for w1, w2, sim in ds:
			if (w1 not in wort_model or w2 not in wort_model):
				print('\t[FAIL] - {} or {} not in model vocab!'.format(w1, w2))
			else:
				human_sims.append(sim)
				sim = 1 - cosine(wort_model[w1].A, wort_model[w2].A)
				if (math.isnan(sim)):
					sim = 0
				scores.append(sim)

		spearman = spearmanr(np.array(human_sims), np.array(scores))
		scores_by_model[wort_model_name] = spearman
		print('[MEN] Spearman Rho: {}'.format(spearman))
		print('==================================================================================')

	return scores_by_model


def test_mturk_evaluation(dataset='wikipedia'):
	print('MTurk Evaluation')
	ds = fetch_mturk_dataset()
	base_path = paths.get_dataset_path()

	scores_by_model = {}

	for wort_model_name in [
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-5',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-5'
	]:
		print('Loading Wort Model: {}...'.format(wort_model_name))
		wort_path = os.path.join(base_path, dataset, wort_model_name)
		wort_model = VSMVectorizer.load_from_file(path=wort_path)
		print('Wort model loaded!')

		scores = []
		human_sims = []
		for w1, w2, sim in ds:
			if (w1 not in wort_model or w2 not in wort_model):
				print('\t[FAIL] - {} or {} not in model vocab!'.format(w1, w2))
			else:
				human_sims.append(sim)
				sim = 1 - cosine(wort_model[w1].A, wort_model[w2].A)
				if (math.isnan(sim)):
					sim = 0
				scores.append(sim)

		spearman = spearmanr(np.array(human_sims), np.array(scores))
		scores_by_model[wort_model_name] = spearman
		print('[MTurk] Spearman Rho: {}'.format(spearman))
		print('==================================================================================')

	return scores_by_model


def test_ws353_evaluation(dataset='wikipedia'):
	print('WS353 Evaluation')
	base_path = paths.get_dataset_path()
	scores_by_model = {}

	for st in ['similarity', 'relatedness', None]:
		ds = fetch_ws353_dataset(similarity_type=st)

		for wort_model_name in [
			'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-5',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-5'
		]:
			print('Loading Wort Model: {}...'.format(wort_model_name))
			wort_path = os.path.join(base_path, dataset, wort_model_name)
			wort_model = VSMVectorizer.load_from_file(path=wort_path)
			print('Wort model loaded!')

			scores = []
			human_sims = []
			for w1, w2, sim in ds:
				if (w1 not in wort_model or w2 not in wort_model):
					print('\t[FAIL] - {} or {} not in model vocab!'.format(w1, w2))
				else:
					human_sims.append(sim)
					sim = 1 - cosine(wort_model[w1].A, wort_model[w2].A)
					if (math.isnan(sim)):
						sim = 0
					scores.append(sim)

			spearman = spearmanr(np.array(human_sims), np.array(scores))
			scores_by_model['_'.join([str(st), wort_model_name])] = spearman
			print('[WS353 - {}] Spearman Rho: {}'.format(st, spearman))
			print('==================================================================================')

	return scores_by_model


def test_simlex_evaluation(dataset='wikipedia'):
	print('SimLex Evaluation')
	base_path = paths.get_dataset_path()
	scores_by_model = {}

	ds = fetch_simlex_999_dataset()
	base_path = paths.get_dataset_path()

	scores_by_model = {}

	for wort_model_name in [
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-0.75-sppmi_shift-5',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-0',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-10',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-100',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-40',
		'wort_model_ppmi_lemma-True_window-5_cds-1.0-sppmi_shift-5'
	]:
		print('Loading Wort Model: {}...'.format(wort_model_name))
		wort_path = os.path.join(base_path, dataset, wort_model_name)
		wort_model = VSMVectorizer.load_from_file(path=wort_path)
		print('Wort model loaded!')

		scores = []
		human_sims = []
		for w1, w2, sim in ds:
			if (w1 not in wort_model or w2 not in wort_model):
				print('\t[FAIL] - {} or {} not in model vocab!'.format(w1, w2))
			else:
				human_sims.append(sim)
				sim = 1 - cosine(wort_model[w1].A, wort_model[w2].A)
				if (math.isnan(sim)):
					sim = 0
				scores.append(sim)

		spearman = spearmanr(np.array(human_sims), np.array(scores))
		scores_by_model[wort_model_name] = spearman
		print('[SimLex] Spearman Rho: {}'.format(spearman))
		print('==================================================================================')

	return scores_by_model


def test_ws353_loader():
	ds = fetch_ws353_dataset(similarity_type='similarity')
	print('Similarity:\n{}'.format(ds))

	ds = fetch_ws353_dataset(similarity_type='relatedness')
	print('Relatedness:\n{}'.format(ds))

	ds = fetch_ws353_dataset()
	print('Original All:\n{}'.format(ds))

	ds = fetch_ws353_dataset(subset='set1')
	print('Original Set 1:\n{}'.format(ds))

	ds = fetch_ws353_dataset(subset='set2')
	print('Original Set 2:\n{}'.format(ds))


def test_ws353_words_loader():
	ds = get_ws353_words(similarity_type='similarity')
	print('Similarity[len={}]:\n{}'.format(len(ds), ds))

	ds = get_ws353_words(similarity_type='relatedness')
	print('Relatedness[len={}]:\n{}'.format(len(ds), ds))

	ds = get_ws353_words()
	print('Original All[len={}]:\n{}'.format(len(ds), ds))

	ds = get_ws353_words(subset='set1')
	print('Original Set 1[len={}]:\n{}'.format(len(ds), ds))

	ds = get_ws353_words(subset='set2')
	print('Original Set 2[len={}]:\n{}'.format(len(ds), ds))


def test_rw_loader():
	ds = fetch_rare_words_dataset()
	print(ds)

	w = get_rare_words()
	print(w)
	print('====')


def test_men_loader():
	ds = fetch_men_dataset()
	print(ds)

	w = get_men_words()
	print(w)
	print('====')


def test_mturk_loader():
	ds = fetch_mturk_dataset()
	print(ds)

	w = get_mturk_words()
	print(w)
	print('====')


def test_simlex_loader():
	ds = fetch_simlex_999_dataset()
	print(ds)

	w = get_simlex_999_words()
	print(w)
	print('====')


def test_msr_loader():
	ds = fetch_msr_syntactic_analogies_dataset()
	print(len(ds))

	w = get_msr_syntactic_analogies_words()
	print(len(w))
	print('====')


def test_goog_loader():
	ds = fetch_google_analogies_dataset()
	print(len(ds))

	w = get_google_analogies_words()
	print(len(w))
	print('====')


def test_msr_evaluation():
	acc = intrinsic_word_analogy_evaluation(wort_model='/Volumes/LocalDataHD/thk22/DevSandbox/InfiniteSandbox/_datasets/wikipedia/wort_model_pmi-ppmi_window-2_dim-None',
											ds_fetcher=fetch_msr_syntactic_analogies_dataset)

	print('MSR Analogy Accuracy: {}'.format(acc))


def test_read_ukwac():
	ukwac = GzipStreamReader(path='/research/calps/data2/public/corpora/ukwac1.0/raw/ukwac_preproc.gz')

	for idx, line in enumerate(ukwac, 1):
		print('[{}]: {}'.format(idx, line))


if (__name__ == '__main__'):
	args = parser.parse_args()

	#test_token_and_vocab_count()
	#vectorize_pizza_epic()
	#test_pizza()
	#test_conll_reader()
	#exit(0)
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
	#test_rw_loader()
	#test_men_loader()
	#test_mturk_loader()
	#test_simlex_loader()
	#test_msr_loader()
	#test_goog_loader()
	#test_msr_evaluation()
	#test_read_ukwac()
	#lemmatise_wikipedia()
	#print('Lemmatising UKWAC...')
	#lemmatise_ukwac()
	#print('Lemmatisation Done!')
	#print('Lemmatising BNC...')
	#lemmatise_bnc()
	#lemmatise_gutenberg()
	#lemmatise_toronto()
	#lemmatise_gigaword()
	#lemmatise_wackypedia()
	#print('Lemmatisation Done!')


	#'''
	#vectorize_wikipedia()
	#vectorize_amazon_reviews()
	#vectorize_ukwac()
	#vectorize_wikipedia_epic()
	#vectorize_bnc()
	print('Running BNC samples...')
	vectorize_bnc_samples(input_file=os.path.join(args.input_path, args.input_file), output_path=args.output_path,
						  cache_path=args.cache_path, current_sample=args.current_sample)
	exit(0)


	print('Running evaluations...')
	rg65_scores = test_rg65_evaluation('ukwac')
	mc30_scores = test_mc30_evaluation('ukwac')
	ws353_scores = test_ws353_evaluation('ukwac')
	#rw_scores = test_rw_evaluation('wikipedia')
	men_scores = test_men_evaluation('ukwac')
	#mturk_scores = test_mturk_evaluation('wikipedia')
	simlex_scores = test_simlex_evaluation('ukwac')

	if (not os.path.exists(os.path.join(paths.get_out_path(), 'wordsim_ukwac'))):
		os.makedirs(os.path.join(paths.get_out_path(), 'wordsim_ukwac'))

	with open(os.path.join(paths.get_out_path(), 'wordsim_ukwac', 'rg65_wort.json'), 'w') as out_file:
		json.dump(rg65_scores, out_file, indent=4)
	with open(os.path.join(paths.get_out_path(), 'wordsim_ukwac', 'mc30_wort.json'), 'w') as out_file:
		json.dump(mc30_scores, out_file, indent=4)
	with open(os.path.join(paths.get_out_path(), 'wordsim_ukwac', 'ws353_wort.json'), 'w') as out_file:
		json.dump(ws353_scores, out_file, indent=4)
	#with open(os.path.join(paths.get_out_path(), 'wordsim_ukwac', 'rw.json'), 'w') as out_file:
	#	json.dump(rw_scores, out_file, indent=4)
	with open(os.path.join(paths.get_out_path(), 'wordsim_ukwac', 'men.json'), 'w') as out_file:
		json.dump(men_scores, out_file, indent=4)
	#with open(os.path.join(paths.get_out_path(), 'wordsim_ukwac', 'mturk.json'), 'w') as out_file:
	#	json.dump(mturk_scores, out_file, indent=4)
	with open(os.path.join(paths.get_out_path(), 'wordsim_ukwac', 'simlex.json'), 'w') as out_file:
		json.dump(simlex_scores, out_file, indent=4)

	print('RG65 SCORES: {}'.format(json.dumps(rg65_scores, indent=4)))
	print('MC30 SCORES: {}'.format(json.dumps(mc30_scores, indent=4)))
	print('WS353 SCORES: {}'.format(json.dumps(ws353_scores, indent=4)))
	#print('RW SCORES: {}'.format(json.dumps(rw_scores, indent=4)))
	print('MEN SCORES: {}'.format(json.dumps(men_scores, indent=4)))
	#print('MTURK SCORES: {}'.format(json.dumps(mturk_scores, indent=4)))
	print('SIMLEX SCORES: {}'.format(json.dumps(simlex_scores, indent=4)))

	#'''
	#test_ws353_words_loader()
