__author__ = 'thomas'
from argparse import ArgumentParser
import csv
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
parser.add_argument('-cs', '--current-sample', type=int, help='current sample', default=-1)
parser.add_argument('-cp', '--cache-path', type=str, help='path to cache')
parser.add_argument('-ef', '--experiment-file', type=str, help='path to experiment file')

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

	print(vec.get_lexicalised_cooccurrences('pizza', use_transformed_matrix=False))
	print(vec.get_lexicalised_cooccurrences('pasta', use_transformed_matrix=False))
	print(vec.get_lexicalised_cooccurrences('beer', use_transformed_matrix=False))
	print(vec.get_lexicalised_cooccurrences('favourite', use_transformed_matrix=False))

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
	for log_sppmi, sppmi in zip([math.log(20)], [20]): #zip([math.log(1), math.log(5), math.log(10), math.log(40), math.log(100)], [0, 5, 10, 40, 100]):#zip([0, math.log(5), math.log(10)], [0, 5, 10]):
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


def vectorize_1bn_word_benchmark():
	from discoutils.thesaurus_loader import Vectors
	from wort.datasets import get_miller_charles_30_words
	from wort.datasets import get_rubinstein_goodenough_65_words

	# p = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews_lc_noid_lemma.tsv')
	p = '/disk/data/tkober/_datasets/1b_word_benchmark/training.tokenised.shuffled.txt'
	bn_reader = TextStreamReader(p)

	# out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors')
	out_path = os.path.join('/disk/data/tkober/_datasets/1b_word_benchmark/', 'wort_vectors')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	# whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_mturk_words() | get_men_words() | get_rare_words() | get_simlex_999_words() | get_msr_syntactic_analogies_words() | get_google_analogies_words()
	# whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_men_words() | get_simlex_999_words()
	# whitelist = get_ws353_words() | get_ws353_words(similarity_type='similarity') | get_ws353_words(similarity_type='relatedness') | get_men_words() | get_simlex_999_words()
	# Bless words
	#whitelist = 'cloak,screwdriver,spade,corkscrew,car,bed,birch,squirrel,cockroach,bowl,apricot,clarinet,shovel,spinach,bomber,cow,beetle,glider,herring,acacia,pineapple,sofa,whale,cypress,knife,cedar,ant,jet,revolver,corn,deer,fridge,stereo,yacht,horse,wasp,vest,missile,tiger,cat,hornet,donkey,snake,turtle,sweater,cranberry,strawberry,elm,beet,gun,coconut,willow,grape,train,ferry,violin,pine,gorilla,lion,table,poplar,axe,bag,fox,tanker,chisel,hammer,cello,mug,lime,alligator,falcon,crow,dresser,dove,sword,oven,saw,rabbit,elephant,cucumber,carp,cod,dagger,spear,butterfly,robin,coyote,villa,bookcase,freezer,grasshopper,cabbage,scooter,helicopter,goat,flute,truck,lizard,penguin,library,washer,bear,tuna,robe,pigeon,bull,vulture,fighter,oak,castle,owl,sparrow,catfish,parsley,glove,hotel,pig,bottle,rifle,plum,coat,rake,wrench,turnip,television,scarf,grenade,goose,eagle,box,cathedral,cannon,lettuce,rat,couch,jar,toaster,blouse,hawk,broccoli,apple,carrot,frigate,peach,giraffe,celery,potato,pear,wardrobe,cherry,cauliflower,phone,stove,trumpet,hatchet,chair,fork,trout,battleship,desk,piano,woodpecker,saxophone,onion,spoon,bus,mackerel,goldfish,moth,pistol,pheasant,guitar,grapefruit,radish,radio,lemon,sieve,musket,ambulance,van,salmon,banana,garlic,beaver,restaurant,dress,shirt,dishwasher,dolphin,swan,cottage,hospital,pub,sheep,jacket,hat,bomb,frog,motorcycle'.split(
	#	',')
	# ML 2010 words
	# whitelist = ['achieve', 'acquire', 'action', 'activity', 'address', 'age', 'agency', 'air', 'allowance', 'american', 'amount', 'area', 'arm', 'ask', 'assembly', 'assistant', 'attend', 'attention', 'authority', 'basic', 'battle', 'bedroom', 'begin', 'benefit', 'better', 'black', 'board', 'body', 'book', 'building', 'bus', 'business', 'buy', 'call', 'capital', 'care', 'career', 'case', 'cause', 'central', 'centre', 'certain', 'charge', 'child', 'circumstance', 'city', 'close', 'club', 'cold', 'collect', 'college', 'committee', 'community', 'company', 'computer', 'condition', 'conference', 'consider', 'contract', 'control', 'cost', 'council', 'country', 'county', 'course', 'credit', 'cross', 'cut', 'dark', 'datum', 'day', 'defence', 'demand', 'department', 'develop', 'development', 'different', 'difficulty', 'director', 'discuss', 'door', 'drink', 'earlier', 'early', 'economic', 'economy', 'education', 'effect', 'effective', 'efficient', 'elderly', 'emphasise', 'encourage', 'end', 'environment', 'european', 'evening', 'event', 'evidence', 'example', 'exercise', 'express', 'eye', 'face', 'family', 'federal', 'fight', 'follow', 'football', 'form', 'further', 'future', 'game', 'general', 'good', 'government', 'great', 'group', 'hair', 'hall', 'hand', 'head', 'health', 'hear', 'help', 'high', 'hold', 'home', 'hot', 'house', 'housing', 'importance', 'important', 'increase', 'industrial', 'industry', 'influence', 'information', 'injury', 'intelligence', 'interest', 'intervention', 'issue', 'job', 'join', 'kind', 'kitchen', 'knowledge', 'labour', 'lady', 'land', 'language', 'large', 'law', 'leader', 'league', 'leave', 'left', 'letter', 'level', 'life', 'lift', 'like', 'line', 'little', 'local', 'long', 'loss', 'low', 'major', 'majority', 'man', 'management', 'manager', 'market', 'marketing', 'match', 'matter', 'meet', 'meeting', 'member', 'message', 'method', 'minister', 'modern', 'name', 'national', 'need', 'new', 'news', 'northern', 'number', 'offer', 'office', 'officer', 'official', 'oil', 'old', 'older', 'opposition', 'part', 'particular', 'party', 'pass', 'pay', 'people', 'period', 'person', 'personnel', 'phone', 'place', 'plan', 'planning', 'play', 'point', 'policy', 'political', 'pose', 'position', 'pour', 'power', 'practical', 'present', 'previous', 'price', 'principle', 'problem', 'produce', 'programme', 'project', 'property', 'provide', 'public', 'quantity', 'question', 'railway', 'raise', 'rate', 'reach', 'read', 'receive', 'reduce', 'region', 'remember', 'require', 'requirement', 'research', 'result', 'right', 'road', 'role', 'room', 'rule', 'rural', 'satisfy', 'secretary', 'security', 'sell', 'send', 'service', 'set', 'share', 'short', 'shut', 'significant', 'similar', 'situation', 'skill', 'small', 'social', 'special', 'stage', 'start', 'state', 'station', 'stress', 'stretch', 'structure', 'study', 'suffer', 'support', 'system', 'tax', 'tea', 'technique', 'technology', 'telephone', 'television', 'test', 'time', 'town', 'training', 'treatment', 'tv', 'unit', 'use', 'various', 'vast', 'view', 'wage', 'war', 'water', 'wave', 'way', 'weather', 'whole', 'win', 'window', 'woman', 'word', 'work', 'worker', 'world', 'write']

	# Tensed Entailment whitelist
	whitelist = 'knead,fuelled,charging,tear,manipulating,sobering,rending,avoided,partner,feasting,mined,shop,blurred,committed,contrast,agree,exercise,evaluated,poured,coercing,deliberated,jigged,run,healed,conserving,substituted,sloshed,sinking,moored,truncheon,slurping,haying,mounting,hanging,pinked,disagreeing,fence,stutter,plug,pulping,cascaded,print,deleting,persisting,cohering,stigmatize,prick,responding,rest,chagrin,visualize,negotiating,thrived,enliven,bantering,terrifying,schemed,reserve,mention,kicking,enraged,knight,scram,toddle,threw,trap,salivating,hoisting,bounce,unbuckle,manipulate,retract,incite,agglomerated,sifting,federated,buy,clack,astound,commercialized,obscuring,cap,reciting,shingling,dip,wetting,foresaw,save,slobber,nosh,diced,sheltering,jolted,stifled,nettle,reconstitute,hurled,bug,boomed,bisect,dazzle,signalled,hunted,chopped,kill,tingle,object,surge,eyed,prophesying,envisaging,goggle,distorting,termed,preen,quiz,muzzle,boated,startled,curse,sidestep,reeking,wading,escort,isolate,refreshed,skyrocket,uplifting,plopped,levitated,robed,deafening,mangle,paving,prospecting,perfecting,rapped,cawed,gauged,staggering,chastise,unbolted,jumping,transcribe,sunk,warehousing,link,mingled,communicate,hoped,strip,air,perking,triumph,wither,dispirited,gaining,leak,ravaged,snort,surprised,lunch,equip,weeding,grasped,baptize,shorten,diversify,twiddled,diagnosing,uplifted,spared,strapping,belch,waltzing,crippling,slug,checking,cleanse,paining,pose,scheduling,core,scouted,bared,serve,voiding,unlace,protected,enamelled,storming,dislocated,rise,appearing,vacating,sorrowing,ravaging,unleash,deploring,branded,design,rue,enticing,intensified,oystering,disperse,tormenting,gab,picked,salivated,wrestling,settle,heaving,decomposed,cherishing,farting,formed,retracted,waltz,overthrowing,won,stirring,scarred,disconcerting,blacken,strain,teeming,deprecating,cabling,retiring,work,cautioning,crash,hear,discovered,leash,exacted,teased,bluffing,intermingle,ornamenting,befell,linked,dared,combining,protruding,store,improvise,conditioning,anchoring,decline,yearning,warble,inverted,overcooking,straddled,crowding,ruffling,joining,evicting,fried,lengthened,molest,chipping,pitting,sow,pronounce,snooping,cooped,admonish,written,roused,kissing,deprecate,abounded,bifurcating,unfold,added,debark,forward,predicted,peering,faulting,outbidding,stash,bridled,cruised,streamed,bored,grazing,crumbled,complimenting,push,quieting,seethed,hosed,wheedled,reverted,stiffened,trimmed,wresting,alternating,exiled,placed,coached,worrying,flouring,following,scar,congratulating,embittering,perambulate,filed,reeked,hailed,reprove,harming,throbbing,starred,fainted,westernized,unbuttoned,chattering,drawling,demanding,thread,yanked,convict,appeased,swoosh,thumbing,tearing,pioneer,outwitting,waiting,resulting,padlocking,generating,raging,sent,hook,compare,tempting,hiccuped,ransacking,zigzag,claiming,professed,tottered,degrease,lilt,devaluing,buttoning,represented,taunting,overreached,produced,accumulating,prosecuting,sloping,slinking,hitched,endured,booking,quivering,neutralizing,impairing,nip,referred,dislocate,leashing,mobilized,catching,overburdening,sprouting,dye,abridging,boiled,waded,confide,gulled,ladle,reclined,tailored,causing,pique,wilt,ionize,shattering,fretted,console,shifted,combing,smothering,swirled,dotted,plotting,listened,lose,salved,roofing,punching,firm,stole,boggled,agitated,corral,beaming,boss,outweighing,steepen,intrigued,thronged,ship,dyed,nipped,crushing,interleave,acknowledging,chopping,imposed,caused,reproved,grazed,whelp,shrivelled,titillate,disentangling,starching,bracket,delight,tidied,express,depressurizing,surging,hopped,blanket,itching,reanimate,stained,beat,unchained,honeymooned,swathe,cycled,bake,equivocated,gilling,canting,sewing,observed,rusting,chirping,choked,satisfied,drape,blinking,translate,sobered,sugaring,tying,dimmed,trumpeting,clipping,praising,tempted,grasping,value,disengage,sequester,chugged,reelecting,ranking,cash,conjoined,disliked,detecting,matching,sliced,dissolving,tape,oiling,corking,championed,quarter,rip,lolling,contextualizing,sacrificing,lending,aging,outraged,defending,concealed,cube,date,knew,challenged,outweigh,befuddling,sniffled,competing,huddle,napping,bay,despised,professing,stupefied,photographing,direct,muttering,rebel,snapping,surrounding,annotated,injuring,generated,preened,lilting,vexing,declare,hobnobbed,hugging,idolize,expire,figuring,jade,imprisoned,eject,stirred,fluorescing,storm,smoked,gathered,flip,potter,pillory,solidify,removing,fixed,flank,wreck,clink,manacled,occupying,budding,hewed,quarrel,brained,thumping,proffered,impel,blending,crabbed,smothered,squalling,stroke,longing,provoke,disgorging,coalesced,pool,hallucinated,tarmac,studied,stating,fatigue,whistling,brawling,allotting,rattling,eat,trusted,formulate,stitched,muzzling,pester,clasping,exported,disorganizing,prune,rearm,rhapsodize,heard,obsess,clouding,conveying,glorifying,revived,singeing,clenching,socked,thwacked,cajoled,lighten,inserting,bloom,honeymoon,waking,cautioned,gorging,associate,sprained,changed,napped,crediting,amplified,harmonized,intimidate,disassemble,tumble,developed,annexing,cuff,wallpapering,prancing,preach,dried,lambaste,fool,disappeared,regard,endowed,slating,embellished,bashing,maintaining,traverse,forbid,ink,tame,forming,pleasing,tailor,outsmarting,rhapsodized,challenge,steady,consecrating,regain,activated,lounging,veered,dripping,raining,thrashed,prodding,roaring,bewildering,brawl,puking,pierced,shrugged,splintered,surveyed,undertake,soiled,dealing,pirated,grieving,overhearing,nail,shuttering,robing,missing,could,constructed,camp,transcending,convicting,incensing,crimped,overheating,living,discussed,clinked,incense,assuaging,coagulated,wrinkle,crumpling,inverting,slandering,sharpen,amassed,banging,rule,cringed,clunking,amused,appall,disturbed,planting,mentioning,bordered,receding,anger,canvassing,loafing,mothering,waggled,clothing,depict,creak,teeter,plating,blot,dawning,payed,anoint,gazing,huddling,cuffing,stopping,scalping,flicked,eroding,anesthetize,overreach,yawned,starch,forgiving,popularize,forfeit,erased,plague,notice,shrunk,exclaimed,resented,covet,exterminating,close,duped,intruded,sagged,frisk,dally,puzzle,conferred,parading,equipped,terming,blackened,emancipated,overwhelm,dithered,confessing,hastening,purring,bombarded,winking,enthrall,suggest,breastfeed,lisping,scorching,caught,upgrading,appraised,enlighten,festooning,summered,exporting,jogged,spotted,billet,sail,inherit,reopening,inscribe,play,strengthened,totter,rejuvenated,contrasted,kiss,enlist,fermenting,transmitted,tutored,strew,interleaving,overturn,parachute,accelerate,bleached,murdered,strained,ruing,quickstep,squeal,roughened,disfigure,mesmerized,knock,flinch,botch,shower,unfolded,undercharged,dazed,subsisted,dismaying,replenish,awoke,affronting,requiring,separating,asserting,calibrating,rabbit,steam,pin,forge,jingling,thirsting,bandaging,mechanizing,efface,rued,leaked,shunned,burgling,dissolved,qualify,watered,perturb,rutting,blinked,printed,chatted,approve,notifying,garland,solder,assess,camouflaging,braised,thinking,beautify,persecute,catnapped,fluttered,chop,propagated,diverging,plough,unfurled,jingled,sidled,cited,torturing,exalting,hinged,insinuate,controlled,merge,long,steadied,stabilized,blotting,police,dying,prostitute,ornamented,entrance,pardoning,traipsed,prevent,bilked,inquiring,pounded,sparkle,enchant,promulgated,humiliate,reel,extorting,lope,taunted,rambling,subsisting,inked,repose,inspect,splashed,intone,corralling,running,dial,shimmered,ticked,stapling,crack,pave,flirt,encourage,bowling,discharged,intertwined,distress,veiled,fizzle,suck,butting,sneered,frustrated,hunger,organizing,slim,discouraged,worried,sharking,wiggle,writing,caddied,yell,changing,grouping,sickened,gouging,disabuse,adhere,ransacked,boxing,moved,bulge,undulated,gash,introduced,loosing,stultifying,whinny,throb,grow,embarrassing,fume,activate,prospering,swooning,tagged,waddle,jarred,knitting,reviling,cried,bicycling,depriving,plaguing,intoxicated,crossing,jiggle,uniting,prevented,fumbling,manifest,damning,scratching,disorganize,disquiet,sounding,abducted,raged,extruded,jail,whaled,destroy,respected,bundle,incinerated,swab,reserved,teetered,elucidated,stitching,debate,slouch,mounded,transmute,zigzagged,invoicing,examined,choking,deduct,fade,reported,sojourning,partnered,lent,nabbed,squelched,expose,incinerating,extorted,delayed,box,transfix,menaced,commenting,passing,devoured,crumple,brightening,asphalting,dispute,bottled,bewilder,cohered,hired,suffocating,accompany,divesting,accepting,leached,transfixed,sacrificed,arise,sharpening,caddying,court,walk,heated,feeing,coop,recommending,dispatching,whack,chose,circling,panic,row,resumed,banishing,stoop,emerged,rushed,collapse,exhilarate,shut,injecting,round,dispersed,strewed,harvesting,mingle,shattered,alerted,outstripping,clasp,opposed,stew,broadcasted,swindle,washed,twisting,polish,imprinting,congratulated,untying,swooshing,devalued,tinkle,predominated,salvage,bloodied,boycott,nuzzled,mashing,fluttering,defamed,stated,kitten,applauded,meeting,slamming,stowing,caulking,muddy,bicycled,trundled,creep,entranced,puzzling,tanning,subsided,heal,describing,hinge,unnerving,quivered,shear,registering,proceeded,swipe,splayed,superimposed,flopping,disabusing,wrecked,wandered,withdraw,splash,powdering,omitting,telex,remit,tanned,exceeded,radiated,waned,meditating,require,depressed,officiated,repossessing,brought,shiver,varnishing,sorrow,desist,unlatch,retching,admiring,lasso,elucidating,hybridizing,leaf,lob,deform,keeping,sliming,photocopying,burl,invigorate,sprout,twitched,snack,democratizing,scramble,posting,refrain,niggling,bivouac,submit,hooking,bugging,vacationed,coupled,summer,jogging,pasting,stowed,reflect,astounding,envisioning,tinting,demand,overthrown,emptying,topping,chuckling,gilding,tainted,plucking,tensed,larding,lounged,clanging,punt,sobbed,walloped,lowered,closing,face,treating,dismounted,decimate,rebuking,slit,unfix,reopen,bristle,eradicating,explained,wallowing,preparing,sponsor,animate,chirruping,regret,succeeding,ban,galvanized,overhanging,weeping,pillorying,burping,sweltered,patter,wrenched,adopting,animating,unionize,widow,grooming,wanting,sing,ushered,denude,grinning,emblazoning,plait,writhe,dove,oozed,stabilizing,pasteurizing,bucking,felt,glimpsed,tamped,chased,attain,append,coalescing,idolizing,blanketed,total,meandered,bewailed,anchor,skate,leavening,telephoned,dubbing,feuded,umpire,decking,decide,skirmished,nibbling,compliment,seducing,hissed,explicate,potting,burbling,poaching,dab,pricing,degreased,parting,gobbled,strike,gauge,sock,outlaw,compromising,puffing,bask,decay,tumbling,locking,lace,caress,shuttled,persevered,sterilize,chaperoned,cruising,blew,lament,deserting,hiring,arriving,recognizing,debilitated,shadowing,disrobe,rile,wetted,fix,reappear,propagating,snip,choreograph,denigrating,admonished,trekked,remember,encompass,depress,pealing,sopping,desecrated,magnifying,peeve,glower,parched,relished,dulled,extracted,happened,grate,lecture,shake,decry,skirted,stretch,cascading,enter,initialling,moult,whimper,attaching,strangling,afflict,designating,crate,file,mourning,drug,concur,disbelieving,resounding,hunch,improving,discriminated,licked,chimed,claimed,permitted,pooled,resigning,calcify,fining,peeking,sidle,leashed,driving,saddle,implied,flushing,gratify,evacuate,reproving,interspersed,overlapping,congregate,inflamed,beaten,ventilating,perused,capsizing,suppose,insinuating,dictated,condemning,void,exposing,chitchatting,evict,redeeming,exhilarating,discriminating,char,manifested,elaborating,perspire,sand,conceding,plop,quaffing,cowing,scrub,rafting,eased,construct,pointing,smile,thinned,crease,illuminating,gamble,consorted,gambol,squeezing,untied,executing,dismay,flame,fitting,urinating,abolished,span,gushing,paved,classed,poked,smother,yelp,abstracting,weekending,chasten,cheat,simmering,confounded,scorning,raced,bottle,exhaling,transmuting,spilt,howling,stimulate,shrilled,carving,unsettle,crossed,pawed,sit,loaded,coiling,repeating,chauffeuring,holiday,speed,talc,baited,carry,floored,guide,stain,ceding,reside,yowl,giving,zest,investigate,blanching,planning,pricked,varying,tamed,scheme,hoodwinking,talking,addressing,obliterated,staffed,hijacked,bussing,blabbing,reduce,limping,initiate,cursing,recover,jutted,fetched,enlisting,encouraged,containing,hypnotizing,coughed,sketch,gurgling,exploded,scalped,glorify,quarantined,nudge,embolden,breaking,chance,peg,maddening,disheartening,gloated,flanked,coach,secure,opened,raze,dedicated,swirling,loaning,obtained,elated,goggled,cite,hover,wandering,excuse,denigrated,confirming,languish,disliking,scavenged,manicuring,scraping,distemper,elaborated,dredged,masking,scent,slopping,scratch,reposed,divorcing,agonizing,pod,maligned,scorch,defrosted,rearrange,ramming,fettered,humidifying,quibbled,ordering,grousing,networking,mailing,huddled,plodded,wincing,boning,expand,squared,impregnate,basked,stem,titter,splaying,entrapped,whittling,repelling,move,exasperating,trolled,riveted,inundating,pound,puckered,severing,shaking,studding,bellowed,guffawed,emblazon,shopping,conquering,interest,incised,frosting,strut,recurring,entangling,objectified,costing,angering,flopped,plot,order,live,appeared,enduring,invested,ascend,shovelled,anesthetizing,blushing,neglect,tie,suppressed,veneered,startling,clobber,swept,filming,drooping,pegged,assemble,rankled,bashed,loosened,grating,stinging,treat,unionized,offer,blessing,glaze,revolt,robbed,furnish,liquefied,pruned,crisping,cherished,hooked,sloped,spray,credit,shadow,dominating,slogged,mizzle,pilfered,distracting,laud,squint,trooping,record,lighted,perforate,persecuting,reproach,gave,crisscross,count,estimating,starting,speaking,creating,immobilizing,wheel,breathed,brayed,nettling,ticket,berth,clamped,flung,hocking,catapulted,disapproved,yearned,bend,beginning,procure,glide,scooping,chink,separate,stomped,chastised,quadrupled,delving,impregnated,burble,warn,patent,plundered,pressured,hosted,whirling,loll,trundle,christening,sob,bop,coughing,tote,infecting,splashing,clapping,deflowering,chewing,buck,eloping,mollifying,bracketing,handled,smuggle,caring,conflicted,deem,assembling,relaxed,stroked,cleansing,expectorated,reinstating,interlinking,enlisted,droning,detail,rationing,remain,sleepwalk,drilled,missed,quieted,cook,biting,skied,adjoined,macerated,fettering,martyred,reviled,persuade,flaunted,repudiate,dazzling,unfurl,razed,measuring,meandering,repelled,rattled,shove,westernizing,lured,confuse,gutted,maintained,grilling,exceeding,planking,twitch,lend,illuminate,mourn,triumphed,classing,bowed,moderated,insist,projected,roiling,traipse,thwack,connected,chronicling,flooding,teetering,gleaming,dazzled,lambasting,tingled,writhed,pardon,lectured,mowed,deprive,floated,divide,witnessing,divest,narrate,telexed,pipping,established,clomped,browned,ascended,dehydrating,swaying,evaluate,rob,protrude,squinting,initiated,notching,dissociating,unhinge,witnessed,distrusted,normalize,published,scream,clobbering,swat,relaying,conspired,comment,saturate,wove,inflating,shutting,reanimated,commingled,extradited,triple,inhaled,dub,acquiring,canonizing,skirting,superimpose,cloaking,bombed,tarnish,explicated,bar,pulsating,arousing,inking,falter,staining,modulated,fuelling,intermix,outrage,raising,affecting,burling,jailed,owning,rewarded,tether,dangled,injured,choosing,knotted,compress,lure,quartering,walloping,electrified,disagree,deter,emulsify,cheapen,viewing,snatched,flinched,proffer,acquitted,dabbed,alarm,diversified,happening,banter,sputtered,banded,tuck,pen,produce,kneel,conceal,intertwining,showing,rose,like,weighted,blister,networked,report,thatching,fazed,calming,team,remained,clearing,turfing,patting,cool,assumed,criticize,modelled,dissipating,enrage,crunched,outsmart,propagate,blend,masqueraded,mutilated,clank,paying,ending,shoulder,fill,dislodging,entered,perturbed,hardening,christen,pencilling,bought,ruled,blinded,coxed,augment,clouded,belted,predominate,freezing,elect,discombobulate,coasting,fearing,misled,crawling,pray,clustered,saw,doused,recalled,hose,embraced,genuflecting,hypnotize,hijacking,restraining,restoring,perfect,star,shuttling,slobbering,defining,encircle,retaliate,strutted,garnished,unzipped,ameliorated,transformed,subsiding,bleach,strutting,gravel,enquire,supporting,conga,implying,jousting,situating,touching,pioneering,seethe,tangoing,deforested,alert,purging,sigh,drizzling,draining,roaming,flaunting,swoon,diagnose,shampooing,warming,searching,click,prod,united,enroll,bait,charge,composing,flanking,resent,trilling,left,eulogized,ski,detected,drifting,insinuated,secluding,offend,qualifying,bleating,convene,murder,fog,embarrassed,spur,fabricating,nudged,tiptoed,decorating,respect,address,clutched,feminized,officiating,sought,devastating,bristling,inspecting,holding,kidnap,levelling,can,teething,pressing,frothed,plundering,eddying,braise,babbled,collect,retreat,cauterized,predict,ate,bugged,stump,crocheting,slinging,splotch,exhausting,shellacking,corresponding,dwell,elevated,fawning,spare,hankered,swallow,loaf,ululating,confirmed,uplift,reprimanded,boggle,milking,jetting,deluge,tinned,unify,house,deposed,turf,explored,fumed,swerved,spooned,chain,decree,settling,mesmerizing,aired,designing,pacified,inspire,exploding,flinching,imploring,crayoned,rotating,meaning,attire,mollified,vacate,visit,hold,rouging,quaking,dash,heeling,degenerated,cadged,harvested,burn,lease,chomping,handle,hugged,draped,plying,condense,disparage,watering,dispossessed,demolish,expiring,matured,crept,bilking,glanced,skittered,dance,schedule,circle,concern,toss,peeked,scale,mused,blink,rejuvenate,characterize,ionized,quietened,scoffing,westernize,welding,caravan,taxied,wedged,hindering,felicitated,celebrating,fleecing,bless,respiring,collated,guarded,peered,understood,dialling,wiped,craning,ran,qualified,squaring,exercised,aggravate,liked,stiffen,copping,asphalt,pull,deforest,enrapturing,civilizing,deify,spice,tugging,straying,feast,rush,asphyxiating,annihilating,enthralled,quarried,shuddered,positioning,discontinue,stealing,investigated,tally,recognize,jumbled,glow,parade,clobbered,miss,forbidding,unlock,scavenge,volunteer,vitrifying,assassinate,commended,airmail,pumping,pursed,harried,cauterizing,combat,contextualize,stink,twiddle,macerating,slope,wreathing,discerned,bewailing,convicted,unleashing,interviewed,trucked,refraining,undressing,transporting,institutionalizing,unwinding,exalted,gouge,diverted,honoring,heap,contribute,permit,crisp,commit,cheating,burgeoning,discomfited,pestered,bargaining,starved,hoe,invigorated,desiccating,surviving,pursing,overthrow,clutch,douse,burbled,sullied,casing,loop,poach,vilified,approach,chlorinated,denouncing,dispatched,strayed,mumbling,drawl,donating,indicated,schmooze,stimulated,coin,emerge,penalized,frame,ticketed,compiling,congregating,desire,birch,reimbursing,deforesting,flamed,announced,burled,snoozed,curved,taxing,deliberating,enthuse,rowed,awarded,endow,cable,slosh,charged,donned,brushing,thudding,gagged,slowed,cashed,banqueting,waver,heightened,incorporating,murmur,shorting,stumble,despoiled,nodded,announce,accumulated,twist,quacking,beguiled,shivering,grumbled,mass,peppering,captained,knifed,zooming,herding,fatiguing,singe,matter,prohibiting,bleeding,firmed,slog,cauterize,canoe,suspended,chase,diffused,appealing,flourishing,sealing,baring,trim,disfigured,omit,hit,claw,hurl,anglicize,abridge,curve,affected,defend,lubricate,woofed,moving,procured,flapped,roofed,complimented,adorn,saddening,restricting,awarding,concede,stagnating,coerce,lumbered,finding,respond,clutter,crackled,depreciated,penning,oust,loosen,titillating,erupt,stroking,enraging,concentrate,win,oscillate,interlaced,connecting,measure,escorted,interchanging,ply,abolish,dissuade,vacuum,mortifying,plumbing,ravish,repulsing,illuminated,veil,articulate,blurt,flowering,glided,photograph,rocket,coat,impeded,photocopied,reheat,spot,gasp,jangle,shortened,clean,crown,pearl,incinerate,awoken,shade,desisted,overcook,spewed,whistled,appropriating,clash,hefted,rumble,computed,weaning,harmonizing,fluster,humiliated,float,deplore,sauntered,vomit,emerging,revering,pestering,loped,supported,battering,woof,showered,cleared,trot,appended,fatten,sounded,puttying,swaggering,install,warping,cremating,inspected,doped,limp,cuddling,impressing,paddle,adored,guaranteed,slackened,degraded,flaring,sever,tack,clerked,angered,associated,spelt,track,traded,damming,bailed,gape,lamenting,duelled,revolve,consent,correlated,ridded,quarantine,hope,warning,speckled,declaring,scribbling,filleted,watch,rowing,peal,conducted,unzip,crashed,germinated,came,seating,exhibiting,awake,designed,crated,guess,enforce,emboldened,gabbing,joined,clomping,crippled,posing,informing,reprimanding,salt,fidgeting,explode,enhance,intoxicating,ogling,spanned,interacting,equalized,flushed,quiet,doubling,clubbed,exulted,bribed,interchange,panelling,devoted,tasting,groomed,executed,informed,polarize,reposing,slink,hindered,wangled,derived,asserted,abused,sojourn,toil,harassed,clutching,manifesting,recruiting,discriminate,corrupting,marvel,dithering,flirting,filling,bullying,rage,trembling,fascinated,scudded,hallucinating,reclaiming,framing,disquieting,butcher,skipper,mixed,wrangled,restricted,spouting,skedaddled,nicknamed,scrape,cementing,attempting,microfilming,nosing,prodded,threaten,attenuate,rally,shaming,spawn,engrossed,roil,stooped,prevailed,lifting,whitewashing,gloved,label,love,telegraphed,rearming,anesthetized,indicting,oxidizing,baptizing,retched,championing,aim,respecting,wakening,wreathe,sniffling,dusted,securing,droned,cajole,demoralizing,corked,tremble,emboldening,quadrupling,excommunicated,theorize,embroider,insisted,shrilling,squished,exterminated,prowling,grin,espied,included,thought,pinion,interpreting,overran,bruising,silvering,implore,colluded,degreasing,dissociated,gyrated,seizing,rutted,flogging,punished,protecting,control,compile,massing,craved,edit,exhaust,wad,warm,inserted,adhering,broadened,disgusted,reassured,twinkling,emboss,yachting,winked,miffed,sheathed,eying,bombing,squeaked,defrost,muzzled,fastened,trickle,leading,resort,shampoo,padlocked,enacted,despair,assisted,departed,breastfeeding,buttonholing,unrolling,buckled,handed,shame,garrotting,mean,smear,hammered,soling,liquefy,garage,glimmering,allure,shingle,aided,hesitated,pout,ooze,burped,littered,obscure,caw,paging,fidgeted,panelled,caramelize,groom,glorying,disintegrating,jig,elected,facing,disembarked,boggling,loose,garrotte,strolled,volunteered,gaze,rotate,squawk,blacking,fizzed,wind,fertilized,liquefying,separated,producing,foretelling,supplied,awing,pucker,choke,hinted,bamboozling,souring,stacking,insuring,erupted,decimating,swear,expunged,reverberate,recollect,owed,decelerating,sole,dispossessing,nursing,debilitating,tangoed,implored,terrorizing,acquitting,sprucing,hurtling,flowed,allowing,instruct,bandage,shaped,enjoy,emanating,deicing,growing,sandpapering,cloak,prowled,plod,harden,leant,thanking,argue,japanned,smoking,scrounging,plucked,slaying,dancing,wrangle,undid,excel,rammed,preoccupy,shrimp,lofting,thawing,expel,bivouacking,faulted,impaired,skulking,blaspheme,spout,slated,stoning,support,appeasing,imprinted,freeze,conduct,worsening,melt,associating,gashing,smelt,risk,whacked,consented,tarnished,disassociated,rustled,terminate,expressed,dream,disclose,ripping,corrupted,content,alarming,floured,mark,kneaded,snivel,mounted,exonerate,stare,desalted,persist,materialize,sweetened,cutting,scolding,desecrate,breathe,committing,chilling,rub,squirt,conciliating,prattling,replaced,hijack,seeding,overstepping,recovering,waving,bandaged,quenched,impelled,phoning,mourned,winnowed,radioed,catch,declined,spattered,ballooned,exhausted,calved,compromise,walking,issuing,charred,invent,rung,caution,curl,shutter,cooed,agitating,blended,nailing,directed,convincing,diversifying,searched,dig,mutate,squirming,cropping,expanded,tiptoeing,dissolve,erase,bewitch,jerk,honor,taper,rubbed,chartering,pedaling,daunting,bearding,fastening,harmonize,undressed,electrocuted,rosin,roasted,publicize,spread,dishearten,tapping,repute,lather,kindle,bound,disburse,mocking,swooned,foaling,unhooking,yoked,envied,skinned,germinate,satisfying,confound,confusing,aching,swagger,snoring,conquered,assist,unbolt,decomposing,deducted,anglicized,catnapping,vying,tin,hobbling,dividing,weed,mope,clasped,exonerating,jamming,blasted,crop,prying,allot,whisk,engraved,perished,curtaining,impress,creaking,stocked,chasing,massacre,lampooning,owing,hungered,sneaking,referring,fashioned,incubated,unwound,gagging,glowering,saluting,clanged,bent,taught,snipped,craving,attained,afflicting,climbed,tweeted,enrolling,canted,assure,reappeared,autographed,posed,bare,heating,suffering,desiring,jiggled,disbelieve,copped,orphaned,persecuted,scuffling,communicated,sipping,loft,confounding,followed,slugging,foam,speared,blossomed,dump,expectorate,offending,shuck,scamper,smeared,shine,infuriated,trapped,buffet,bestriding,peruse,remunerate,theorized,glinted,evening,scouring,rambled,provoked,worked,diminishing,blab,taxi,capsized,checked,zipped,blossoming,binding,plunking,promulgate,surmounted,introducing,mock,agonized,canvass,slide,hobbled,blitzing,inject,pawned,energized,spank,chitchat,convulsing,tootle,denuding,zipping,model,bricked,hoed,institutionalize,schlep,paged,answering,unburdened,deluging,yapping,turned,pocket,toured,quickened,guzzling,sputtering,captured,conquer,insure,liquidating,squinted,engraving,categorized,steamed,swilled,coexist,galling,swapping,stencil,distinguished,dine,stud,reeling,shedding,hoist,supposed,desinged,write,convulsed,breakfast,covering,extort,enriched,curried,stub,aggregated,patrolled,crushed,sponsored,shelling,herd,stamp,casting,slithering,vaulting,intersected,wanted,slither,stopped,snarl,correlate,appointing,hurry,stooping,clenched,shunted,eating,licking,towel,improvised,paddling,promising,whistle,vibrate,speckle,sensing,spitting,wobble,banged,papering,confer,fading,sighted,scolded,cloaked,demagnetized,poking,clomp,censured,carrying,assembled,trundling,wolf,assail,confessed,camouflage,perfuming,smashing,pulling,bench,varnished,helped,intermingled,conveyed,mowing,distressing,prance,funnel,diminished,cawing,rebelling,battling,functioned,steered,exiling,sugar,remunerated,blanch,castigated,rallying,tattooing,experience,undertook,leased,charring,flashing,thudded,sighting,sprain,detonate,broiled,stubbing,narrated,carbonized,page,paste,tunnelled,bribing,deluded,feathering,raft,don,contriving,littering,enhancing,asked,guarantee,hailing,insert,muse,wallpapered,wheeling,snapped,consuming,skewed,plotted,evading,soak,conked,despising,pass,knelt,normalized,conjectured,tick,stashed,zinged,cooperated,enriching,transcribing,cheered,growl,awe,whirled,uproot,raiding,dined,revolving,dealt,tripled,convert,misspelling,emasculated,zinging,electing,arguing,sully,decorate,winced,closed,weaned,emanate,scattering,hankering,brewed,fine,mate,abutting,winkled,adore,met,tagging,heightening,assassinated,devoting,adjoin,peddle,bluffed,incise,excising,invoice,defaming,sicken,secularize,creasing,clapped,assume,increase,referee,squealed,decrease,emancipating,disengaged,lilted,tax,colluding,annotating,sloshing,piping,intended,persuading,dampen,tootling,echoing,wiggling,stupefy,drummed,constructing,cheapened,intoning,sledge,staging,walked,limited,plant,assault,harrying,tootled,doodle,bucked,confided,slammed,hating,trick,spell,hinging,bunted,sandpapered,barring,churn,mottle,counted,harness,relieved,tow,denigrate,rummage,easing,desert,integrating,lied,dong,ripped,groaning,pulped,surmise,noshed,appalling,anguished,thrummed,seek,experienced,recovered,screwed,breastfed,remitted,punted,vacillating,confiscating,battled,salve,obscured,roosted,pulsate,mailed,nabbing,analyze,assuming,patched,brew,tour,belting,cease,shucking,soaked,ease,promenaded,tightened,gazed,nutted,sweeten,inspiring,overlap,travelled,convening,banning,burgle,thrum,denting,betting,surrendered,galled,smashed,copied,arrange,incited,drying,delighted,tarried,surmount,required,crowd,baffling,discomfiting,gratifying,ferret,gawk,banished,block,wrestle,overcame,discombobulated,pomaded,interleaved,tolled,sip,enthralling,blamed,prosecuted,stemmed,excommunicating,articulated,convinced,studded,tugged,masquerading,ambling,spoiling,aggravating,curving,consorting,stood,wail,attempted,cajoling,allocate,borrow,intend,vilify,designated,umpired,spurred,freeing,leavened,cycle,obsessed,apportioning,abrade,evacuating,chatter,recollecting,haggling,reaffirm,buffeting,unrolled,tucked,establishing,clove,persevering,pasture,recalling,memorize,commingle,disabused,honeymooning,bisected,smudging,boxed,splutter,sapping,liquidate,swearing,lounge,complaining,gladdening,ingesting,judged,piquing,humidify,surrounded,dipping,squirmed,return,consulted,trimming,muddied,shooing,puttered,scheduled,abraded,whinnied,wished,darkened,pipe,abated,lisp,certified,interviewing,seized,tasted,bobbed,sell,studying,formulated,organized,proceed,hand,uttering,swish,coping,train,igniting,bail,sold,arresting,preceding,pulverizing,chugging,itch,nauseated,dismayed,strapped,fogging,outraging,abating,circumvented,crucified,improved,exclaiming,tolerate,giggle,pouted,pining,blast,bully,disclosing,soil,paled,accelerating,burrow,complained,flower,immersing,tossed,coddle,appraise,result,quieten,embarrass,recoiling,coasted,touched,intimate,barred,loafed,fermented,bricking,saturated,batted,mincing,mewing,concoct,screeched,permitting,daze,clay,wriggle,uttered,leap,sunbathing,gulped,distempered,scheming,suspecting,succeed,rotated,intrude,priced,colliding,adjoining,purloined,dawn,alerting,groping,coring,film,abusing,papered,powdered,liberated,identify,grunted,compared,extended,quack,project,promoted,happen,brief,slew,sew,stifle,clucked,shelved,battle,dicing,credited,milk,rate,repressing,veneer,revolted,swaggered,jesting,bark,fetter,foretold,frowned,wiring,amazed,scrawl,immunizing,tired,reorganizing,caressed,hum,twining,deafened,disappearing,invented,shunt,excise,unfasten,expounding,humming,clammed,gabbed,analyzed,pocketed,hobble,foretell,inveigling,billed,martyr,impeached,nicking,recasting,joke,exceed,venerate,crochet,acquired,vanish,shrouded,stumbled,skirmishing,bothered,electrify,granulated,commanded,jarring,garbed,slashed,encircled,charm,suggesting,saddened,cased,infuriating,liberate,abducting,inciting,lampooned,toasted,scared,sizzled,doubting,starched,saying,skating,martyring,pulverized,prosecute,minting,splatter,relishing,dilute,sneezed,rid,padded,amass,bed,whisper,accumulate,furnished,displeased,bludgeoning,slurp,bracketed,cellaring,munched,spicing,quiver,ejected,marinated,alternate,waning,pencil,pad,soothed,spindle,outwit,shrill,hosing,oscillating,esteem,storing,reelect,doctoring,drum,reduced,bronzed,coddled,annoy,cry,reeled,niggled,faltering,chattered,chiding,baking,roared,lightened,quenching,freed,emasculating,beam,disseminate,quartered,escorting,prostituted,bearded,spear,focussing,conversed,accrued,snooped,detailing,beautifying,lettered,treated,inform,envisaged,spar,moderate,waggling,fluctuated,yawning,tease,fatigued,crooned,marinate,seething,harnessed,slack,parching,limit,plastering,withdrawing,tantalized,diminish,lacquering,jabbed,entangle,induced,lisped,censuring,note,piled,buckle,depend,ticking,simmered,disintegrate,trade,forested,arching,contracted,curled,abbreviate,heading,swatted,contrive,masticate,sipped,larded,reproaching,wedge,craned,amusing,shouting,baiting,pale,inflate,willed,eluding,masticating,overstep,beating,exulting,putting,botching,corroding,elucidate,aiding,fold,arming,bunting,teethe,butter,clanking,misting,shrug,flabbergast,open,battered,identified,wilting,publishing,basking,affirm,moisten,invoiced,ravage,overturning,scalp,peck,evaporate,chirp,screech,spook,detailed,retrieved,administering,scalloped,shrieked,state,latch,spurted,blinding,affronted,watched,swig,scrimping,wrinkling,corrode,revolved,topple,civilized,transmit,ride,darken,stabbing,harm,togged,replenished,delaying,mound,shrimping,panicking,seep,urged,garnishing,whipping,incorporated,lacquered,retaliated,draw,salving,berthing,dispirit,deepening,tempt,exile,plate,teamed,abasing,yacht,uprooted,coating,remaining,clogged,glue,remarked,wagering,attack,preening,staffing,divorced,kissed,roost,compelled,sop,elating,tip,mantle,formulating,repossessed,motoring,commend,lulling,scrambled,muddling,radiating,earned,withdrew,eschew,perfected,tan,erode,crowed,packed,castigating,gasped,went,burying,polymerize,entering,borrowing,shining,ruminating,transfixing,agreeing,tamp,articulating,tattooed,see,shipped,deviating,eddied,prayed,garb,overwhelming,puttering,throng,loved,case,ramble,thumb,partnering,criticized,foal,meet,scrutinizing,deterring,memorizing,dehumidifying,moped,acidified,claim,ranged,jibing,broke,buttonhole,delay,sway,revere,deciding,focussed,demobilizing,softening,transmitting,pastured,labouring,scare,modernized,spying,straggling,scoff,cabled,rounding,conferring,assert,shovelling,forwarded,appointed,overcharging,arm,dredging,galvanizing,consolidated,weighting,concerned,consumed,monitoring,asking,deviated,repossess,gnaw,summering,benched,ploughed,trek,holidayed,moralized,emasculate,publicized,crocheted,roast,gathering,outlawed,curry,squish,spattering,rifled,enquiring,stinking,grunt,dodge,bin,skimmed,stammered,deposit,name,stucco,contort,sighed,hissing,board,thrusting,sunbathe,lessened,situate,accelerated,told,crumbling,swung,intending,shucked,pecked,disagreed,interact,sidestepping,nickname,sequestering,tickle,typing,squall,objectifying,depicted,lumber,sidestepped,carpet,twittering,tinkled,hazard,macerate,braying,combating,skirt,calibrated,fashion,dismembering,hanker,persuaded,vaporizing,diverged,swopped,absolving,obligated,whitened,mash,dousing,passed,yelling,trickling,veined,amaze,tarry,nursed,render,ionizing,reap,threaded,blitz,petrify,cage,repeated,tell,jigging,choreographed,slandered,spoke,execute,blush,drip,festoon,tweeting,beget,haggle,sprawled,entrapping,stunned,beeped,alienating,dock,spatter,ululate,wheeled,embroidered,stashing,eventuate,degrade,considering,chanced,confederated,staple,unite,vary,lasted,indicating,including,unbutton,befall,patrol,dart,pleased,remembered,shimmering,figure,condoned,neigh,entrancing,grabbing,dictating,blatted,knotting,spruce,crowded,develop,combined,sleepwalking,banned,proposed,antagonize,twitter,distort,exit,minted,lie,donning,purpose,printing,compose,stuttered,skin,castrating,returning,replying,bubbling,increasing,lard,tantalize,glaring,deterred,picking,knighted,espy,cooked,oxidized,scrubbed,forgave,resume,barter,coerced,estimated,grinned,stagnate,lopped,strengthen,calcified,nicknaming,snooze,spanked,squatted,refunded,deleted,plied,pounding,hacking,consoling,satiating,scampering,leed,term,unlatching,travel,embossed,chrome,immersed,protesting,parachuted,insult,electrocuting,planked,chaining,jibed,applaud,dissent,silvered,spluttered,steeping,ding,extirpated,squawked,foaled,wiggled,democratized,decompose,contaminate,scowling,deliberate,derive,riding,motivating,hurting,ogled,barging,bothering,totalled,materializing,thatch,perturbing,bludgeoned,dotting,widening,protect,varnish,transforming,swigged,impair,enlivened,pop,glowing,carolling,flustering,wilted,signing,hoodwinked,sting,pomade,firing,buffeted,whittle,conjoin,stunning,ridiculed,flatter,excoriated,abhorring,slumped,obtain,salute,reek,carried,blackmail,ameliorating,breaded,yowled,boycotted,belt,fizzling,plan,spraying,jutting,travelling,bud,nipping,circumcised,mottling,dehumidify,glinting,siring,froth,punch,squeaking,decorated,dammed,fossilizing,strolling,deceived,adding,flinging,rotting,reserving,heaped,lapped,consider,placated,sighing,vulcanized,chronicled,lodging,fizz,approving,coaching,treasuring,raised,naturalize,gnashed,cough,tacked,ridiculing,staggered,certifying,waltzed,differentiated,fumble,shave,maturing,mottled,straggled,fart,allocating,gumming,scold,fancy,butted,exonerated,cleaving,disgorge,dropping,bewitching,wowed,discourage,behave,daring,thrilling,consoled,explaining,petrified,buried,training,padlock,saddled,etch,stabled,dwelt,docking,lightening,dissuading,pay,hush,prosper,buzzing,probing,specified,reversing,enervated,promised,unburden,dimming,soled,managed,pirouetted,disdained,rolled,wadded,molt,embitter,conversing,eluded,burgeon,hunched,gutting,repudiated,reuniting,peppered,nauseating,snaring,rocking,anointing,bragging,harass,immolate,precipitated,tumbled,paint,swell,prevail,hovering,fractured,deepened,fracturing,rasp,tossing,obligating,wedging,torment,blackmailed,precede,desalt,cleansed,naming,inducing,spew,envying,roof,lingered,glittering,wheezing,flashed,enlightened,mentioned,contaminating,wielded,champion,robe,rinsing,reprimand,grooved,bivouacked,lifted,playing,sundering,going,sleighing,croaking,suffocate,expunge,forged,reconstituting,pulsated,include,bartered,admonishing,reverberating,hollered,bump,remonstrated,brawled,demanded,swerve,sailed,dehydrate,discussing,refund,burrowing,resounded,wireless,wired,prize,attacking,crimping,quoting,croak,immunize,contrived,guard,commissioning,drowsing,graduated,sung,abstain,buttering,surround,pine,excite,rescue,stomp,flooded,snorted,enjoying,slumbering,equalize,interpreted,stabling,adorning,care,humidified,purchasing,squash,denied,stapled,diving,reassure,reacted,killed,gummed,seduce,imposing,leave,rolling,imagining,humble,troop,clunked,skiing,clamp,collating,noting,argued,defended,lashed,skated,ruminated,buckling,buzz,port,linger,treeing,blared,mocked,relinquished,pouting,unscrewing,ossified,altered,heave,fenced,apostatize,shout,duck,snuff,quitted,undercharge,envisioned,garnish,bubbled,boogie,trust,ululated,narrowing,lunching,flipping,appealed,joust,rebelled,administered,mine,borrowed,settled,nickel,whining,squabble,hiding,aiming,representing,chiselled,harnessing,mulched,retreated,shoot,awaking,debug,reassuring,varied,tile,chartered,bear,menacing,embalm,glowed,treble,outsmarted,trucking,jolting,booming,strewing,swap,cost,washing,reputed,darting,foresee,botched,prepared,unfurling,snickered,reelected,flatting,ring,treasured,wowing,snared,released,bomb,voicing,smiled,scrabbling,fondle,vanishing,broadcasting,scour,jested,shouldered,drained,remitting,enslave,rankling,feather,deceiving,accuse,whine,disseminating,balanced,saved,scrabble,moaning,pummel,strap,part,categorizing,threading,distill,diapered,exude,snickering,imitated,tarred,excited,bellowing,delete,oxidize,emptied,introduce,inflaming,looking,piped,wrangling,activating,damaged,punishing,commenced,gurgled,scoffed,degas,peel,destroyed,needed,fattened,emulsified,jabbered,shortening,abhor,dated,parboiling,sickening,ravished,rusted,twinkled,polishing,removed,shaded,ascertain,telephoning,minded,crashing,rubbing,revitalized,contract,dipped,perplexed,loathe,rewarding,stitch,muting,affiliating,buttonholed,jeer,imprint,scowl,recede,yellowed,rave,discarding,flicker,snipe,overrun,radio,sharpened,donate,hacked,victimizing,stuccoed,mixing,wiping,acidify,restore,stormed,disassembling,camped,completed,recreated,chew,unhook,warp,sandpaper,funnelled,perishing,plagiarizing,thrilled,boring,beach,diapering,felicitating,atrophied,clothed,multiplying,affront,exchange,popping,hollering,ascending,fetching,contextualized,forced,gaped,canonize,smash,substitute,impeach,shaving,scrubbing,dislocating,refresh,unbuckled,partition,crumble,returned,collide,thumped,roughening,reclaim,hiss,rekindled,screened,bloody,dribbling,talk,shock,inebriated,competed,accompanying,enraptured,collecting,clanked,add,pasteurize,alienated,brighten,slipped,berrying,democratize,diffusing,disturb,quench,roam,cosh,welcome,entertained,doodling,grab,rejoice,rebuked,subsist,releasing,straining,rearmed,jangling,thrill,sucking,descending,scraped,corresponded,rouse,writhing,intermingling,chastening,leaching,purge,neck,blasphemed,preoccupied,recruit,umpiring,corrupt,landing,exchanged,grounded,smell,blare,reanimating,congregated,visiting,irk,patch,behaved,chirrup,erasing,wadding,strangled,conjuring,wax,tethered,flecking,wallop,voided,steepened,tango,appease,unfixed,venerated,sculpting,swishing,misspelt,coil,nab,sidling,crunching,invade,bobbing,fossilize,jingle,prop,jangled,paralyze,flow,ferreted,splicing,fabricate,wallow,condemn,thunder,marry,embezzling,even,reddening,avoid,vault,filmed,pointed,fashioning,taped,relay,arose,lessen,led,exiting,oared,tapered,amazing,sowing,raving,guessed,bickered,beeping,singed,pinned,caramelizing,network,sponged,bridged,blanketing,branding,captivated,supplying,clang,pruning,suffused,pricking,winnow,spanking,guffaw,ruffle,lead,expound,demobilized,shovel,scowled,abound,lived,chucked,revitalize,teasing,unburdening,depleting,sweeping,tinkling,chill,is,dare,partitioned,annoying,offended,laced,line,sunder,starring,maddened,het,handling,form,specifying,blindfold,dispiriting,stalling,selecting,lolled,tunnel,grilled,blazed,exasperated,underlie,stage,puff,devote,rumpling,outnumber,enlivening,publish,whispered,kicked,cubed,clump,tormented,deflating,fray,shambling,sleigh,madden,ration,maim,effacing,picturing,abetting,patching,boil,signalling,crush,evaporated,enervate,enrapture,sift,dispatch,severed,depositing,ordained,fell,mortified,partitioning,cringing,drooped,growling,duel,yielded,leasing,annihilated,release,enchanting,sputter,flood,japan,dither,doff,spawned,warehoused,hail,catapulting,excavate,grumble,swooshed,protested,immolating,swatting,rifling,riveting,arranged,flattened,batter,traversing,cede,annotate,worm,asphalted,coarsening,babbling,chained,tucking,enlarging,participated,exterminate,ossify,demagnetize,remarking,ported,mutating,hide,oiled,drawing,multiply,drumming,crystallize,yellow,sprinkling,crewed,mill,welcomed,howl,jaded,refunding,cackled,branch,divert,demagnetizing,shearing,whitening,abut,respodning,erected,relying,motivated,puckering,rushing,acclaimed,want,shellacked,punting,mortify,segregate,coining,relax,toppling,froze,create,citing,collapsing,peeping,dapple,enticed,crinkle,describe,ferried,purifying,consolidating,clap,dirtied,occurred,climb,deplete,resulted,concocting,blabbed,reproached,ironing,blame,interweave,percolate,outweighed,encompassing,nestling,fossilized,trawl,tethering,disarm,connect,pup,swelter,contain,copying,rooting,mangled,named,mining,spill,doubted,sniffle,extraditing,double,interlocked,plunked,comforting,looped,disparaged,advancing,sprinkled,occurring,traced,glisten,bamboozled,perforated,proliferated,pupped,sobbing,looked,controlling,crackling,select,corroded,atomizing,evade,upset,certify,piloted,pain,disassociating,equivocate,overcoming,tiring,sucked,bronze,post,attenuated,feeling,adoring,engendering,slept,crisped,monitored,rafted,dumping,fleeing,gesture,swelling,bunt,wore,crating,troubled,perk,slaughter,dismissing,waxed,terminating,injected,reckoned,perform,bridle,placate,listening,ringing,belched,meditated,soiling,captivating,carbonize,auditing,bamboozle,showed,fencing,combine,hinting,plank,gill,sauntering,lubricating,molting,dressed,moaned,necking,sheathing,creaked,pressuring,tripling,funnelling,nationalized,reclaimed,raided,unscrewed,enquired,disrobing,irritate,befuddle,beckoning,conking,assailing,slashing,cub,take,hushed,rebuke,generate,screeching,fighting,jumped,deflowered,shipping,focus,awed,bet,thud,disputing,plunge,repulsed,inflated,gambled,patted,stamping,delved,detach,evolving,stacked,leaping,misappropriate,flocked,criticizing,scalded,puttied,manipulated,burdening,fawn,appropriated,lumped,procrastinating,tore,objectify,blurting,fault,regaining,taxying,vomiting,assaulting,rend,nested,forked,chlorinate,suggested,understudying,consecrated,immobilize,purchase,pronounced,hung,sniped,tilt,stopper,expected,empty,rap,hummed,typed,depopulated,approached,halted,wriggled,alluring,strode,packing,convince,wait,hybridized,prattled,retort,reverse,linking,log,slitting,dissociate,survived,billing,distinguishing,awakening,impregnating,astonish,suckling,confess,pilfer,esteeming,nesting,burgled,ask,metamorphosing,tickling,land,winning,enrich,urbanized,bleaching,zigzagging,doffed,sopped,recounting,whish,defined,metamorphose,crowning,gleaning,notched,rouge,spy,scuttling,slowing,wearying,tree,gull,dismiss,clunk,deported,thrust,outmatch,castrated,gush,toted,skulk,enacting,polling,editing,reflecting,mashed,dozing,redden,decried,drenched,participate,pierce,clawed,build,sleet,plaiting,remunerating,pinging,peeved,excavating,rekindling,described,frown,splitting,sequestered,slacked,chortled,discern,appraising,excusing,lettering,demonstrate,mangling,niggle,unionizing,tarmacked,cull,scrawling,damage,tarnishing,pumped,nestled,quoth,cabbage,rattle,dozed,honored,demolished,disappointing,rearranged,resonated,hunt,patenting,maimed,popularized,cowed,lauded,snowing,place,coddling,race,levitating,constraining,tram,risked,think,sitting,blow,roughen,unscrew,wish,distend,infected,demolishing,reply,slay,unpegged,hug,deteriorating,pause,capsize,packaging,pedaled,overcooked,answer,pitch,flitted,churning,coaxing,declining,lick,wire,abounding,lapsed,catapult,stir,shape,nose,initialled,pinking,powder,engaging,impelling,lashing,eschewing,unnerve,crooked,curdling,defecate,molesting,blurring,abbreviating,lopping,provoking,reaping,heartened,swindled,outwitted,thundered,chiselling,intimidated,faze,wheedling,confederate,mutilating,preventing,realizing,salvaging,depart,padding,porting,cemented,twittered,attaining,reiterated,whip,entertain,iodize,pinning,scaled,rumbled,clung,languished,aged,repulse,bumping,regaling,bossed,shuttle,overlapped,venturing,enjoyed,amplify,resonating,soaped,buttered,envision,trod,gyrating,smirking,dreading,iron,directing,issue,blooming,tinted,decided,haunt,charcoal,jerking,ridding,respire,untie,vacationing,perceived,fit,modernizing,perforating,surprise,acknowledged,grant,whelping,imbued,distribute,reaffirming,venture,perch,punish,firming,marking,shuddering,abuse,admired,resign,fitted,navigating,cleave,gnash,irked,worsen,forwarding,sculpted,blest,fled,debilitate,working,recall,creeping,milling,moping,detesting,rot,mystified,berthed,recited,asphyxiate,snow,poisoning,paper,seasoning,preferring,skulked,squirm,voiced,charmed,investing,constricting,say,ducked,retire,cheer,massed,haggled,bundling,rendezvousing,banish,swaddle,strengthening,slacken,inveigled,itched,invest,loaned,drowsed,sculpt,replacing,wade,jabbing,hurried,garaging,swaddled,teach,counselled,thumbed,narrating,rated,rain,channelling,distributed,mutilate,shrivel,interrelate,knit,munching,boned,croaked,exhibited,pursuing,unwind,thickening,marvelling,tracing,border,heralded,beckoned,inhaling,revelled,rotted,dusting,stultified,foresting,coarsen,confiscate,observing,exuded,blotted,has,burgeoned,lamb,drowse,slumber,surged,flit,improve,contorting,microfilm,reckon,pip,remonstrating,bloodying,tweet,debugged,slacking,swill,escape,stress,civilize,flicking,oblige,judging,rendering,pair,prostituting,curling,initiating,clambered,destroying,feud,search,lampoon,singing,unhinging,resting,march,yammer,drench,drink,gorged,arch,confabulate,flap,rank,detect,ambled,jabbering,bellow,reckoning,shriek,slap,reimbursed,grooving,ousted,ventured,comb,sour,blossom,frolicking,ossifying,guffawing,acidifying,pitying,soften,gloating,festered,frustrate,conflicting,talked,commercializing,channelled,fretting,beamed,trumpeted,binning,grasp,skipping,knighting,destabilize,canonized,sight,crave,packaged,seeded,nationalize,cocked,repaid,bowl,anguishing,seduced,gild,wavering,bantered,nationalizing,moralize,mothered,snore,valued,differing,pile,blushed,motor,imbue,shelve,melted,piling,petrifying,crew,straighten,hopping,incising,pilot,snuffling,understand,cuddled,butt,condensed,grudged,murdering,hiked,tinging,tutor,sense,lurching,allow,stemming,revered,instructing,pirate,scatter,intersperse,thaw,deformed,cowering,style,shingled,furrowing,find,refer,slump,hauled,lurched,treed,intersect,clinking,taking,explicating,frowning,gabble,wormed,steaming,cuddle,levelled,ostracize,trudge,fizzled,yoke,labelled,reheating,sink,dashing,reviving,abase,mutated,swayed,hewing,plunging,cured,sorrowed,induce,seed,lurked,misappropriated,hiccup,nodding,reddened,wet,contributed,primped,acquire,biked,boasting,coincide,owned,heighten,wrecking,willing,evicted,tussled,tingling,barged,disconnect,poke,appropriate,interlink,condone,reach,proving,quitting,straightening,nosed,tramping,deeming,emancipate,lurking,picnicking,evolved,departing,congratulate,robbing,indicate,flirted,horrify,retaliating,toting,naturalizing,sling,flowered,calk,stimulating,string,drowned,constrict,marvelled,sinned,carpeting,relish,chuckle,flaunt,veneering,tousling,atomize,grill,drown,intensify,feminizing,roll,quicken,bursting,ameliorate,inveigle,detaching,quarry,reclining,encompassed,ensued,performing,revealing,situated,smack,trudging,sojourned,distrusting,wipe,gall,dubbed,shamed,tramped,speak,slunk,squirted,proclaiming,nominate,retracting,grated,loitered,pinged,nick,scuttle,landed,drugging,seize,wheeze,modernize,skirmish,titillated,kneading,bicker,correct,perfumed,rim,react,yelping,hitch,smearing,surmounting,thrashing,desisting,despoil,slouching,asphyxiated,prohibited,recollected,prompting,dreamt,flew,mature,retired,waste,test,quaver,scurrying,sizzling,ignited,vilifying,slip,tied,purred,snatching,rallied,group,milled,affiliated,computing,reimburse,clamping,unbolting,sniffing,stagnated,cut,dot,despoiling,incorporate,withered,revitalizing,panicked,resuming,filched,consult,evolve,shading,discharging,evaporating,glimpsing,hire,blistering,propping,rhapsodizing,ripple,mute,winding,figured,denounce,journeyed,defrauded,revile,recording,differentiating,trouble,wonder,redeemed,swarming,displeasing,halter,taunt,commanding,overhung,shadowed,forcing,burp,litter,residing,enhanced,shopped,sparred,jump,paralyzed,secluded,insulting,breakfasting,outnumbered,snipping,solidifying,bayed,circumcising,hurtle,croon,turn,whacking,tallied,quibble,winter,crouched,damned,pot,hike,showering,mechanize,couple,swabbed,standardize,collate,vibrated,sending,leaved,ache,assimilated,collapsed,precipitate,tricked,imitating,intersecting,readied,dawdled,intruding,cloister,chatting,guiding,abet,gawking,overrunning,host,snicker,dread,nestle,approved,installing,coexisted,antiqued,participating,mislead,freshening,raid,ticketing,advising,beard,caning,flexing,whitewash,drooling,disrobed,slumping,devouring,fooled,reorganized,marching,scaling,screamed,administer,commingling,biking,sponge,entangled,confederating,disdaining,classified,headed,die,brush,clumping,incubate,breakfasted,ensue,spearing,curdled,charter,persevere,reaching,gulp,skewing,helping,preceded,straddling,standardizing,pack,chilled,recompense,contented,trotted,trammelled,calving,assaulted,deporting,slam,tracking,abandon,negotiate,displease,scorched,leer,lash,laminated,festooned,bond,saving,restrained,captain,relieving,witness,impeding,skedaddle,overwhelmed,filtering,mother,staring,allowed,resenting,trace,chirped,nap,rock,dredge,bargain,whipped,insulted,healing,shrinking,moistened,molted,shaping,bonded,bread,paused,backbite,pressure,dismantle,whiten,flitting,gut,portray,riddle,stigmatized,dissatisfy,nibbled,wafted,embossing,recur,saddling,look,flickered,exhale,loving,spinning,probe,rippling,woofing,exchanging,vulcanize,diffract,groaned,bang,stewed,tassel,blaspheming,sambaed,wane,feasted,desired,throbbed,plunder,knifing,dribble,redressed,weaken,apportioned,obliterating,scanning,clear,spooking,cherish,limped,resorting,proclaim,panel,sponsoring,humiliating,last,disgorged,aimed,popularizing,rinse,greasing,concerning,poisoned,drain,justifying,alarmed,capped,struck,donated,substituting,grimacing,dissipated,liquidated,animated,stun,bestride,tolerating,quavering,whooped,clucking,streaked,babble,sanction,pent,pour,survey,schmoozed,advised,steeped,quaff,watching,turning,daunt,eschewed,labelling,soldered,embroidering,smacking,lost,viewed,bartering,yapped,conditioned,equipping,wailing,recline,change,embrace,snuffled,bumped,captivate,scrimp,clashing,wondering,gouged,cox,clicking,equalizing,dwelling,materialized,mix,pat,illustrated,pawn,improvising,guaranteeing,thwacking,whisking,command,gibed,export,retch,glimmered,receiving,teaching,gilt,growled,deducting,victimize,swallowed,subside,admire,reproduce,interrelated,vibrating,flocking,descend,quarantining,terrorize,allocated,commissioned,selling,constricted,pilloried,inheriting,dug,sniggering,antagonized,shocked,riffled,housing,pirating,remove,gambolling,socking,skimming,sober,disassociate,constrain,cheeping,lancing,decelerate,differ,trawled,blocking,lock,spoon,laminating,perceive,slithered,patented,enslaved,pulp,check,moderating,faded,spruced,portion,dickering,approximate,affirmed,wavered,integrated,grease,boycotting,luring,gratified,banding,cuckoo,jug,bequeathed,lassoed,waddling,bailing,toddled,fear,step,swarmed,spied,ruminate,bustling,teaming,evened,enervating,wrote,bringing,cracking,perspiring,peeled,trusting,twanging,destabilizing,foamed,realized,recruited,incensed,imitate,regularize,arrested,twiddling,abstaining,coagulate,wrung,sapped,relinquish,absolved,shroud,nauseate,winnowing,dive,reorganize,shouted,moo,splice,splattered,lower,disseminated,join,upsetting,regaled,forage,chronicle,begetting,sensed,shepherd,darkening,bedded,register,believing,framed,polarized,woke,drizzle,feel,pluck,sponging,drill,adhered,acclaim,split,parted,pinch,degenerate,compressed,dehydrated,journey,guzzled,smooth,insured,vacated,lined,celebrated,mow,scoured,tunnelling,barking,disarmed,patrolling,paralyzing,bottling,spin,imbibe,photocopy,sneak,felicitate,translating,pearled,stay,exact,harry,stroll,zoomed,whelped,unhooked,top,boasted,fuel,sanctioned,bonding,balance,wagged,anointed,mellowing,ascertaining,cast,poll,preaching,entrap,capture,crimp,pawning,encouraging,dissuaded,invert,disgraced,inscribing,smarten,soothe,twitching,luxuriate,engendered,leafing,illustrating,squashed,surpass,snarling,transfer,stalled,justified,force,hoot,spun,eliminate,consulting,nutting,completing,pith,stencilled,distinguish,begot,arouse,primping,salivate,coped,sacrifice,affirming,rating,grace,sprayed,crooking,baste,hoodwink,meander,trill,blunted,snored,repel,hurling,clerking,gasified,toast,burst,bending,gobble,resound,flabbergasting,wishing,veining,doctored,loading,surrendering,scalding,filing,yammering,intimated,bundled,smoke,surrender,losing,diluting,proceeding,disheartened,shellac,sneer,scudding,inventing,camping,outmatched,relieve,chaperoning,conjecture,rekindle,reviewing,grabbed,seated,stow,lusted,skinning,jolt,echo,caramelized,converging,dismember,romping,eliminating,yield,gracing,imply,plagued,flare,damaging,exist,scoot,belittled,purified,paraded,collude,bestrode,transpiring,drugged,chip,energize,swirl,bowled,tinning,mutter,dicker,praise,commented,manufactured,dabbing,recorded,clawing,redressing,oscillated,burden,stared,stringing,squeak,spoilt,pinched,pipped,suckled,emulsifying,ridicule,roasting,squeezed,bewail,gibing,rinsed,poison,mew,point,shelled,snarled,attacked,swigging,disturbing,smiling,eliminated,enthused,bequeathing,come,speeding,defrosting,fleece,annexed,victimized,jammed,intoned,shrivelling,ceased,lobbed,negotiated,steep,deport,butchered,acquit,intertwine,shrouding,stripping,thundering,cling,weighed,trading,coveted,screen,deposing,genuflected,canvassed,panting,owe,disgracing,neutralized,expect,shooting,notified,reacting,panted,pelting,assured,bedding,feminize,kidnapping,slaughtering,squirting,functioning,shrieking,replace,dismembered,repress,splotched,scandalizing,dismissed,plummet,prefer,characterized,grumbling,refereed,bus,poached,pegging,scavenging,plunged,deride,whimpering,worsened,pity,wave,pitied,dice,uncoiled,stumbling,grunting,magnify,rejuvenating,reverting,drydocked,follow,sanded,present,annoyed,beautified,dawned,raved,submitting,fluctuating,sanding,traipsing,mechanized,affiliate,receded,dirtying,measured,shambled,dispersing,promote,warbled,prospected,wheedle,berry,masticated,yelled,sin,dyeing,rupture,treasure,inundated,flock,befuddled,crisscrossed,forfeited,officiate,doctor,tortured,tailoring,snuck,trailed,jailing,orphan,lambing,ascertained,flattening,defecating,eulogizing,knocking,head,regularizing,fret,riled,bruise,transported,filleting,ruffled,stultify,revealed,whittled,tolling,portrayed,proved,integrate,haunting,touch,painted,go,nibble,shackling,toddling,tried,bash,help,delivered,frolic,sizzle,counsel,ransack,enforcing,nudging,tilting,selected,beached,repeat,decreased,prickled,swished,leaning,scooting,salted,overawe,clacking,compensate,caged,switching,spurt,fattening,moistening,heat,antiquing,straddle,drizzled,federate,fancied,ripened,spooked,transmuted,hoing,contributing,contorted,cripple,deprived,heralding,extracting,racing,despise,cork,perching,concurred,extirpate,folded,groan,reopened,dope,widen,stammer,fowl,assessing,clubbing,burr,spool,incriminated,debated,compelling,slackening,amalgamated,bagged,stuttering,immerse,forking,slipping,muttered,dining,duping,eddy,griping,overcome,delve,planted,devastated,emanated,baptized,conducting,tattoo,rivet,mellow,camouflaged,weep,agglomerate,fazing,nest,purse,burdened,teethed,kindled,compromised,grew,cleaning,slogging,daunted,depressurized,malign,scald,suspect,squat,depopulating,grafted,concentrating,disbelieved,hurtled,trebling,threatened,tussling,greet,chortling,deluding,purr,degenerating,swiped,calibrate,knife,injure,pressed,mail,fleck,fumbled,shred,moralizing,debarked,disquieted,backbiting,blindfolding,mesmerize,contrasting,visualized,pulled,issued,venerating,impressed,splattering,reviewed,wailed,disembarking,obtaining,hammering,dressing,fillet,overreaching,pitching,disconcerted,unsettled,portraying,betted,wash,distracted,triumphing,broken,fall,brain,phone,joked,accused,trickled,plaster,basting,subtract,seclude,recognized,pulverize,crewing,accompanied,gripe,bulged,excavated,lecturing,dispossess,snatch,retrieving,drifted,daub,autographing,dissipate,sniping,brand,awaken,antagonizing,outbid,club,clustering,revive,wrinkled,endowing,admit,dry,curdle,pelt,lacquer,piercing,mince,merged,deafen,apportion,steadying,picture,wake,bother,gashed,testing,kneeling,calumniate,cringe,peeling,humbled,falling,housed,paddled,root,stoned,peer,flour,projecting,droop,dislike,eulogize,overstepped,desiccate,binned,sound,barge,gloried,eventuated,occur,outstripped,recoiled,stretching,rake,charming,dust,scratched,thin,belittling,enthusing,smoothing,moor,abandoned,lugging,yammered,scandalize,chauffeured,distended,scout,extruding,gained,stand,standing,opposing,doodled,defraud,yoking,forbade,overheated,graduate,adorned,quarrelled,soared,struggled,perambulating,confiding,stab,mooned,nailed,dallying,scented,vitrify,cuckolded,haunted,drenching,hefting,gloving,mooring,secured,wed,irritating,surpassing,worshipped,presenting,ventilate,played,caressing,pouring,hoarding,reinstate,smudge,fogged,ruined,prospect,prized,profess,blocked,range,ingest,conjure,discombobulating,pinioned,pattered,expounded,gleamed,wobbling,shorted,ruin,airmailed,lunched,hearing,lump,survive,tiling,unifying,narrowed,rocketed,glistening,blackmailing,elope,unnerved,prowl,sniggered,gladden,wearied,conjoining,wink,grouching,class,conflict,filter,advance,carol,rode,compensated,slimed,unbuttoning,smarted,cadge,augmented,languishing,ferrying,gulling,pictured,moan,dented,bubble,consecrate,shell,abetted,phoned,crisscrossing,pacify,voted,digging,stall,beware,extradite,plow,deprecated,regretting,squelching,lurch,misleading,chlorinating,lamented,loosening,strangle,parboil,gulping,modulating,hayed,contaminated,skyrocketed,rescuing,hazarding,vaporize,cropped,polished,propped,commence,vote,interlinked,elate,hobnob,denounced,had,abstracted,jam,burnt,striding,shun,simpering,overhear,elaborate,interpret,perish,goggling,desalting,portioned,flapping,fizzing,tussle,diverge,forest,ceded,condemned,glove,simper,plodding,knowing,fixing,lathered,spending,kept,shod,succeeded,supposing,circumcise,invaded,proliferating,thronging,undoing,armed,bled,infect,shoving,bow,declared,gobbling,balancing,dangle,scanned,begin,analyzing,coast,waved,sprawl,bustle,own,cowered,toboggan,upbraid,weary,scribbled,lift,bleated,signal,lengthening,bribe,capping,chalk,died,hearten,accept,gestured,spiced,attempt,tug,shackled,castigate,trudged,assessed,underlay,logging,needing,twirled,detonating,pepper,naturalized,discard,embalmed,snacked,failing,outlawing,kick,sifted,stored,ploughing,disengaging,dampened,stayed,appalled,absolve,loom,cultivated,rationed,fainting,drydock,inflame,handcuffing,clogging,crow,halting,precipitating,cremated,clattering,clothe,piqued,loping,hitching,trail,dinging,end,reading,expired,conserved,ferment,vanished,marinating,bicycle,set,freshen,superimposing,preached,recount,mop,primp,tire,sparing,recreating,stigmatizing,riling,unbuckling,lop,sagging,broaden,engaged,cluck,sewed,bleat,wringing,astonished,procuring,taste,haul,chagrined,jar,fracture,elevate,overburden,spoil,attired,identifying,mosey,unchain,deflate,fling,stomping,chide,flush,concurring,ping,yap,squashing,swindling,motorize,wasting,deluged,wagered,misspell,wakened,filch,cladding,thicken,roamed,excoriating,zip,pried,interlace,flutter,annex,cultivate,omitted,deriding,floating,peddling,worship,rupturing,disentangle,buttoned,canoeing,shoved,coupling,trooped,continued,cooling,exited,communicating,apprenticing,crabbing,explore,sup,dumped,tinged,dazing,cloistering,decrying,hesitating,tar,pollute,terrify,elude,immunized,discontinuing,nurse,believe,scintillate,created,seeped,edging,depended,ruptured,straightened,wasted,smoothed,minding,baked,laughing,delude,radioing,squatting,disconnecting,prepare,depleted,fowling,wagging,scallop,lug,urging,breathing,snap,conjecturing,slung,bronzing,surprising,whitewashed,bloomed,salting,stalk,dissatisfying,fascinating,redrawing,pirouette,recast,embezzled,brightened,contracting,sprouted,cram,refreshing,wedding,giggled,purposed,distilled,enlarged,clinging,voyaged,coagulating,solidified,cashing,jest,scrimped,suppress,insisting,soar,butchering,radiate,taxed,maiming,warned,cellar,thriving,billeted,balloon,widowed,rendezvous,learning,building,whinnying,moseyed,squishing,taming,pasteurized,jet,pry,heel,squabbled,amalgamating,prickling,complete,rely,dripped,navigated,kindling,gladdened,vie,puked,gushed,existing,eloped,slid,apprenticed,found,normalizing,upbraided,dirty,horrifying,furrow,deny,shatter,spotting,overhang,regained,fetch,cluttered,topped,hovered,dedicate,dislodge,hungering,lusting,censure,feathered,undress,distilling,swim,gibe,mint,screaming,letter,bridge,lobbing,styled,repaying,wince,wrested,dodging,descended,waited,shivered,initial,dangling,hoping,chromed,shot,murmured,tarrying,tighten,garlanding,samba,plated,recoil,cramming,illustrate,harvest,streaming,unlatched,correspond,served,telephone,bopped,confused,prattle,blaming,lingering,damn,spewing,yellowing,thrive,covered,reheated,tag,crying,placing,satiate,riddled,suffocated,unpeg,flex,continuing,crouch,crane,decreed,jerked,quacked,snacking,melting,forfeiting,obsessing,stupefying,soaking,noted,broadening,towing,swarm,registered,thank,furnishing,disappear,screwing,inherited,reconstituted,fired,drift,stressing,blurted,browning,ruling,opening,redraw,undercharging,chime,leaven,wrenching,saturating,imprison,silence,instructed,spelling,drinking,carve,gesturing,bite,praised,coined,inlay,fined,wielding,hunting,responded,discontinued,scorn,thawed,wreathed,bluff,worry,chiming,snorting,fasten,quarrelling,repay,adopt,scrambling,bathing,tittering,reproduced,devastate,cuckolding,disgruntle,disbursed,depreciate,weaving,transferring,erect,extract,squawking,pirouetting,weld,withering,worshipping,dickered,cudgel,granting,behaving,manufacturing,incriminate,painting,relayed,inebriating,deposited,wallowed,nominated,blind,cow,deteriorate,hove,culling,seasoned,soaping,courted,greeting,cream,dim,converting,suffer,mopping,befalling,rendered,inquired,rimmed,reiterating,scaping,sag,cluster,cock,eradicate,blistered,slouched,divested,herald,deforming,accruing,pummelling,tense,tapped,skittering,sparring,pained,coax,mating,rumpled,hastened,immobilized,hew,matched,pined,braising,decreasing,calve,cackle,mount,neaten,misappropriating,schlepped,depreciating,hardened,smartening,laminate,estimate,clipped,sowed,sanctioning,despaired,drooled,stepping,smudged,lanced,switch,christened,disclosed,announcing,spooling,quit,steering,televised,paid,coinciding,condoning,plagiarized,interlock,mobilizing,granulate,denuded,faltered,slumbered,riffle,decimated,assimilate,griped,mist,lasting,believed,knocked,rendezvoused,ogle,contained,compressing,handing,ushering,sprinkle,transferred,examining,redress,silencing,destabilized,federating,pronouncing,slow,rimming,flog,transcribed,toiling,interlocking,assuage,scrounged,underlying,prohibit,clashed,considered,diffuse,slobbered,chinking,shrugging,twirl,categorize,allured,imagined,plowed,glance,swerving,utter,filled,sheltered,mushroomed,slice,arising,throwing,regretted,agonize,deal,dating,please,herded,notch,isolating,tainting,commending,consort,hasten,fondled,proposing,chat,visualizing,sniffed,waggle,bob,bring,ranging,thinning,silver,churned,acclaiming,applauding,bowing,intimidating,stippled,designate,brushed,embellishing,embellish,debating,cooperating,muddled,posted,creaming,spit,pedal,corralled,clam,laughed,disembark,inhale,chomp,stripped,solace,sterilizing,avoiding,scampered,deriving,grinding,circled,infuriate,gleaned,distorted,decentralized,lean,pump,ended,speeded,brewing,weigh,extend,flickering,intermixed,bathe,boating,raise,mind,mooed,kidnapped,classifying,inlaying,clacked,replenishing,said,drove,disintegrated,astonishing,cheep,skip,twined,monitor,perplex,autograph,quarrying,berried,nut,jibe,erupting,audited,crucify,appoint,suffuse,relate,enamel,ached,smacked,explain,liberating,gallop,relinquishing,docked,pairing,doping,frustrating,rousing,lap,despairing,boarding,sundered,slicing,navigate,vulcanizing,loan,degrading,frolicked,towed,stumped,teem,propose,chanted,doddering,floor,built,pitted,stretched,stray,gyrate,shredded,obliterate,gaping,sleeting,sniff,boat,sunbathed,hobnobbing,frosted,modulate,polled,grip,weave,refrained,urge,annihilate,shoo,ground,twanged,seat,loitering,stung,transcend,clambering,calcifying,galloping,glancing,unpegging,atomized,crinkling,compel,unroll,affect,glint,lighting,vex,sadden,rasping,broadcast,demoralize,pushed,elevating,pant,electrocute,clatter,reflected,expelling,embittered,smartened,enact,skyrocketing,galloped,load,stratify,dribbled,prophesied,provided,convulse,frightening,inscribed,terrified,booked,delighting,tarring,deified,celebrate,ordaining,envy,perceiving,motored,comfort,read,pasted,gripping,represent,stop,bone,scarring,risking,planned,afflicted,chant,bat,abridged,cooing,dinged,consume,policing,magnetize,garaged,orphaning,grouch,yawn,disappointed,lying,ferreting,piloting,calling,decayed,made,light,reducing,skipped,discerning,clamming,manicure,slash,glistened,doze,faint,sneeze,perched,examine,cavorted,stumping,pawing,keep,demobilize,touring,musing,delivering,attached,inquire,creased,burning,petted,spouted,disbursing,stock,choose,dialled,rocketing,demonstrating,smelling,persisted,exhibit,fabricated,purged,engage,holidaying,yelped,forging,ordain,whined,coveting,grouse,show,depending,bisecting,labour,effaced,knitted,prompted,bifurcate,excelling,ladling,glimpse,listen,sweltering,rescued,trailing,smirk,vitrified,benching,stable,cause,fertilizing,cooled,dashed,disgrace,tensing,translated,presented,frost,exhaled,tricking,bore,protest,ostracizing,lengthen,apprentice,captaining,extirpating,mulch,unfolding,copy,oyster,put,plumb,assisting,disarming,undo,gasify,curtain,quizzed,gawked,clattered,imbuing,squeeze,regularized,glazing,salvaged,capturing,limiting,shampooed,warmed,carved,hocked,shed,mooning,convened,draping,wriggling,flared,depicting,striking,cocking,circumvent,harassing,predicting,ordered,inspired,broil,schmoozing,jiggling,reversed,germinating,inlaid,crook,engender,proliferate,bragged,loosed,nuzzle,agreed,enchanted,abstained,distressed,necked,crinkled,relied,submitted,deliver,regale,groove,pinching,cluttering,cooperate,caulked,steal,concealing,admitted,related,unhinged,anguish,devour,bequeath,drawled,short,spend,resigned,yielding,cane,meditate,bombard,rustling,commiserate,notify,conserve,dictate,combed,brag,soothing,gnawing,molested,flabbergasted,wag,bulging,deemed,putty,condensing,condition,scurry,suspending,televising,puzzled,guessing,appeal,whimpered,scan,consenting,plagiarize,brick,abstract,muddying,pummelled,wedded,free,cycling,recreate,hated,beaching,suckle,faced,reporting,leaking,frighten,appending,pasturing,jog,chug,reveal,provide,branching,cored,toppled,outnumbering,rent,loomed,ornament,aggregating,metamorphosed,interwove,sterilized,mumble,flowing,dissenting,lodge,glared,scud,disconnected,retreating,roar,bussed,shift,interchanged,toughened,unlaced,smirked,echoed,acted,compensating,nicked,tobogganing,hoard,enlarge,looping,took,unlocked,suppressing,coated,bridling,tiptoe,slugged,ventilated,smuggled,decelerated,sailing,ejecting,bristled,bathed,etched,received,approximating,swing,promoting,pitched,immolated,prickle,chuckled,exuding,veiling,braining,ceasing,courting,installed,consolidate,retrieve,flaming,rummaging,prevailing,mingling,discovering,irritated,truck,stubbed,pet,shredding,plummeted,canoed,grind,coalesce,conk,finish,upbraiding,blindfolded,dallied,grimaced,polluted,desiccated,chalked,establish,perfume,swiping,guided,valuing,bopping,latching,judge,assigning,cracked,drone,mopped,justify,overheard,unsettling,traversed,jeering,squelch,marked,imbibing,snowed,press,composed,tiled,invigorating,chanting,resorted,arched,enforced,excelled,ostracized,whale,handcuffed,quadruple,bridging,dampening,percolated,streak,bordering,confirm,scandalized,devalue,logged,blaze,lambasted,longed,lining,streaking,boom,splintering,doubled,shoe,deteriorated,crystallized,plummeting,munch,disdain,readying,baffled,urinated,serving,age,pealed,holler,diaper,abased,tarmacking,strung,flourished,rust,skedaddling,pattering,cuckold,secularized,totalling,trying,receive,flooring,picnic,edge,struggle,commission,disentangled,conjured,pocketing,grappled,scooped,troubling,crawled,discuss,stuck,expressing,sealed,flop,water,fleeced,overburdened,graze,peddled,scorned,commercialize,pushing,spluttering,spellbind,interview,dismantling,gambling,paired,approximated,sleepwalked,drive,razing,snubbing,stocking,graduating,allotted,humbling,snuffle,appreciated,voyaging,know,trotting,undulate,wintered,widened,gag,dupe,crystallizing,converted,emblazoned,stabbed,finishing,shaved,hid,cavort,trapping,tint,hesitate,lulled,telling,overcharge,curtained,leafed,sugared,appreciate,chomped,caddy,screening,thrumming,level,freshened,fire,recurred,disorganized,jousted,mobilize,objected,jabber,earn,spraining,tightening,culled,interested,clicked,scrutinize,rained,abduct,advanced,revert,upgrade,exposed,tacking,wield,knot,guarding,dissented,assigned,fertilize,peek,rounded,defame,scurried,specify,exercising,satiated,shone,subtracting,glittered,mystify,cheapening,neatened,quake,operated,parboiled,institutionalized,picnicked,differentiate,joking,rumple,gum,learn,starve,tottering,blasting,grafting,view,discarded,wintering,swimming,lacing,remark,scuffle,cascade,tipping,divorce,grieved,festering,slopped,quote,fooling,blanched,polymerized,straggle,discouraging,ensuing,hint,suffusing,bill,tolerated,vacillated,duelling,rumbling,wept,edited,reunite,manicured,admitting,daubing,dull,magnetizing,hoarded,trebled,plopping,wrestled,simmer,wring,exhilarated,regarded,skitter,rustle,scrutinized,prove,rut,bawling,aggravated,drilling,tousled,called,japanning,looming,foreseeing,gurgle,arranging,alternated,assign,scalloping,nobble,cuffed,telegraphing,relaxing,diffracted,batting,killing,paling,toll,minced,genuflect,seeing,vomited,bargained,belching,loathing,fester,enlightening,jumbling,hitting,reiterate,cheering,shackle,ingested,crackle,gleam,define,disparaging,scuttled,glued,milked,tidy,crucifying,jab,rippled,thatched,hiking,wow,punched,whooping,decaying,tittered,welded,gloat,snubbed,steepening,disapproving,liking,fluoresce,disgruntled,stammering,toasting,feed,mounding,quavered,commencing,choreographing,cloud,lugged,parachuting,penalizing,channel,shuffling,interesting,dismount,menace,blat,surpassed,vaporized,diagnosed,narrow,seeping,lofted,creamed,cavorting,undulating,depose,disillusioning,book,moulting,plonked,blunt,cackling,trawling,glimmer,toughen,maligning,manage,compiled,saunter,deepen,shelving,diluted,shunting,suspected,scrawled,pelted,mated,deflated,threatening,alter,denying,trolling,neatening,broiling,counting,audit,snigger,stifling,semaphore,scintillating,imbibed,promulgating,shepherded,magnetized,subtracted,staged,shrink,dominated,boiling,graft,glowered,teemed,married,gather,incriminating,eroded,compute,uprooting,quibbling,bag,plunk,shark,chinked,astounded,shimmer,scrabbled,chucking,lurk,unzipping,scuffled,prophesy,answered,fuming,calm,soldering,quietening,steer,inebriate,fought,trained,puffed,penalize,terminated,theorizing,dreaded,pausing,give,bifurcated,understudy,unleashed,flustered,disassembled,gliding,hypnotized,fight,resonate,greased,reinstated,briefing,enslaving,loiter,entice,iodizing,toot,flecked,clench,coexisting,cope,disgust,oozing,burrowed,gnashing,oar,plastered,engrossing,desecrating,envisage,surveying,convey,detonated,correlating,accepted,riddling,study,caulk,crouching,publicizing,skewer,imprisoning,shepherding,leaving,sweep,banquet,exult,drool,bruised,abandoning,modelling,slander,rooted,jeered,transpire,revel,positioned,nominating,clad,tasselled,perspired,bounced,reappearing,quickening,airing,cross,pecking,collaborate,whispering,organize,voting,ducking,hang,wangling,starving,impose,flatten,frightened,bludgeon,screw,policed,need,coincided,snoozing,disgusting,sneering,recommended,turfed,waddled,repressed,bilk,excoriate,lumping,rasped,budded,hurrying,flogged,nettled,lapse,erecting,decentralizing,exclaim,slurped,grouped,jut,repudiating,debarking,muted,hunching,scrounge,warehouse,boarded,encircling,rearranging,held,pooling,eye,hatching,thirsted,season,doubt,aid,quaffed,weeded,hurt,restrict,grappling,eradicated,heaping,troll,hoisted,obliging,weakening,latched,tracked,bray,drew,vacuumed,retorted,charcoaled,purloin,wean,shuffle,lull,heeled,advise,noticing,concocted,interlacing,multiplied,fry,undertaking,evaded,preferred,frayed,commiserated,locked,trumpet,collaborating,detached,concentrated,squabbling,raked,obliged,muddle,drank,grounding,discover,waxing,indicted,exalt,sap,noshing,flexed,bouncing,galvanize,baa,hazarded,energizing,clog,polka,vied,exciting,wheezed,twirling,debugging,plowing,televise,blazing,manufacture,popped,incubating,excommunicate,neglecting,embezzle,tipped,bawled,reaped,fluctuate,lauding,cohere,curing,increased,signed,swelled,correcting,intimating,dodged,shocking,flick,spent,pink,band,microfilmed,prospered,rummaged,coxing,murmuring,confining,restored,predominating,evacuated,sleep,distract,plumbed,abbreviated,flash,stipple,sheared,lassoing,detest,amplifying,expanding,conciliate,heartening,sliding,chalking,gain,foraging,tooting,thirst,hatched,fancying,groused,fly,appear,kennel,augmenting,seeking,quizzing,dropped,baying,fraying,boast,gilled,swore,plonk,stalked,styling,roosting,struggling,quaked,caned,thanked,chisel,agitate,climbing,sketching,stream,laboured,shunning,wolfed,shouldering,switched,cleaned,spooning,surmised,grudge,meant,vein,maintain,setting,acknowledge,resided,ironed,unified,chastising,tapering,frying,promise,probed,bearing,collaborated,scenting,shuttered,crowing,hybridize,enrolled,purify,thump,welcoming,lowering,ravishing,hinder,sprawling,spurting,ballooning,try,slate,cop,hauling,lessening,garlanded,will,toiled,glitter,assailed,conspiring,snuffing,warped,stone,sat,shooed,dreaming,classify,spreading,stunk,currying,groped,dappling,stewing,exacting,ferry,graced,eaten,ripen,crumpled,collected,staff,hock,exasperate,startle,coo,moon,recompensed,mushroom,yanking,stressed,billeting,ready,zoom,etching,pursued,swinging,frothing,began,tramp,detested,hooting,pilfering,lathering,stack,fidget,taint,sullying,motorized,pardoned,extrude,grieve,saluted,aggregate,appreciating,relating,pranced,crammed,placating,tutoring,bike,start,tread,seal,reaffirmed,catnap,dress,hack,torture,electrifying,sign,spilling,supply,clerk,isolated,inundate,beguile,wrest,dissatisfied,pearling,cheated,promenade,prompt,gasping,disconcert,supping,clumped,skippered,remonstrate,whirl,cement,disillusioned,jumble,soured,standardized,peeped,brown,glorified,yowling,sired,attenuating,interspersing,branched,preoccupying,chided,memorized,hate,objecting,entertaining,divided,wander,intensifying,bustled,neglected,ruining,revolting,intrigue,luxuriated,spindled,protruded,forgive,tamping,swallowing,cremate,expunging,tested,bawl,gorge,schlepping,sire,tap,bleed,altering,belittle,percolating,snare,stamped,vexed,lubricated,operating,function,embracing,sparkled,ranked,started,speckling,pottering,shifting,winkle,dislodged,journeying,ladled,make,developing,warbling,accrue,unlocking,depopulate,assuring,snoop,discharge,foraged,sinning,remembering,shoeing,crimson,lapsing,flying,rouged,splinter,scouting,amuse,segregating,lapping,glory,doffing,bewildered,fork,unfastening,veer,wear,snuffed,usher,flipped,wager,invading,boogied,wound,suffered,tickled,telegraph,bagging,scattered,expectorating,comparing,hosting,motivate,spawning,combated,voyage,fondling,demoralized,dominate,levitate,ripening,hiccuping,antique,merging,nod,pacifying,spellbound,neighing,swaddling,slop,deifying,volunteering,filtered,redeem,weight,wobbled,stepped,spurring,defecated,polarizing,depressing,glare,evaluating,abrading,stratifying,breading,price,bury,toughening,amalgamate,perplexing,drop,coaxed,splay,danced,interweaving,sticking,intriguing,canter,gossip,rejoiced,miff,distributing,constrained,continue,indict,daubed,chortle,grope,furrowed,beckon,reached,wangle,stoppered,scribble,marched,tallying,cellared,decentralize,dulling,earning,square,distrust,defrauding,button,disputed,twisted,chancing,conceded,cadging,granted,performed,yearn,confine,frisked,drowning,pioneered,edged,marrying,differed,chewed,photographed,wondered,cloistered,praying,outstrip,coming,weakened,valet,tog,carpeted,briefed,atrophy,mask,mystifying,barked,sparkling,mooing,uncoiling,chauffeur,glazed,thickened,laugh,deice,endure,providing,flourish,reverberated,converse,sketched,chipped,proffering,abate,fawned,ousting,handcuff,bewitched,rankle,commiserating,dawdling,giggling,making,impeaching,caging,recounted,iodized,sledged,shamble,smart,coiled,mumbled,masked,regarding,satisfy,restrain,skim,wallpaper,obligate,unfastened,scoop,approaching,pit,amassing,stagger,buzzed,intermixing,rove,aroused,buying,accusing,uncoil,weekend,call,leapt,excised,act,horrified,transform,hammer,dam,replied,vaulted,crooning,snub,irking,fructify,mantled,perusing,pencilled,howled,shudder,feuding,finished,abhorred,cure,crab,experiencing,leach,roiled,alienate,characterizing,diverting,ting,spellbinding,purchased,urbanize,gauging,shelter,reproducing,disappoint,whished,crawl,reward,supped,rocked,softened,throw,frisking,occupied,extending,folding,dedicating,wafting,match,observe,dent,abutted,harmed,twine,decreeing,greeted,understanding,procrastinated,polluting,filching,prizing,calmed,position,voice,lust,realize,gluing,purloining,shook,recite,dawdle,stratified,attach,smuggling,spanning,rested,blaring,equivocating,portioning,urbanizing,overcharged,tilted,conspire,whisked,depressurize,sweetening,pursue,jetted,trembled,transport,contenting,tooted,moseying,puke,vacation,rejoicing,hay,refereeing,shuffled,chaperone,expecting,slapping,crayon,challenging,operate,potted,corrected,procrastinate,rising,deplored,hushing,urinate,romp,existed,lance,perked,trammel,intoxicate,nuzzling,abolishing,soap,rapping,demonstrated,learnt,terrorized,glean,worming,reunited,understudied,acting,guzzle,revelling,upgraded,stride,silenced,stabilize,masquerade,trekking,proclaimed,impede,tantalizing,expelled,black,basted,baffle,wrench,squealing,neutralize,whaling,slaughtered,break,luxuriating,pick,arrest,blur,spliced,soaring,riffling,confiscated,crowned,award,counselling,castrate,darted,taping,deck,occupy,spooled,package,mollify,stick,romped,sneezing,putter,beguiling,raking,wearing,dappled,comforted,bombarding,dreamed,collided,cared,disapprove,arrived,mellowed,anglicizing,suspend,mulching,massacred,confined,blitzed,swabbing,converge,embalming,engross,complain,cruise,recommend,send,bossing,peep,managing,awakened,manacle,investigating,massacring,decked,veering,fascinate,secularizing,bickering,crunch,grimace,clip,scaring,converged,transcended,skippering,overawed,engrave,waft,foaming,anchored,mushrooming,compete,fail,type,stalking,derided,hooted,sovietize,dismounting,visited,slapped,treading,wolfing,arrive,heft,dodder,amble,grapple,dismantled,assassinating,assimilating,misted,review,stippling,thrash,discomfit,twinkle,oppose,lumbering,rifle,vacuuming,deflower,disillusion,cultivating,trilled,imagine,overheat,assuaged,towelling,twang,interacted,lodged,beep,clamber,grudging,deserted,cooking,paw,atrophying,tousle,bullied,chastened,idolized,hop,whoop,zing,overturned,chuck,oil,swam,excused,adopted,magnified,surmising,hatch,deceive,boogieing,hallucinate,blowing,smarting,retorting,waken,disfiguring,sleeping,sheathe,failed,cover,ram,feared,coarsened,weighing,staying,circumventing,petting,loathed,yank,tidying,vacillate,blunting,esteemed,flatted,transpired,segregated,scooted,parch,deviate,stiffening,swilling,noticed,promenading,flee,exploring,ignite,cower'.split(',')

	print('Word whitelist contains {} words!'.format(len(whitelist)))
	import math
	for log_sppmi, sppmi in zip([0], [0]):  # zip([0, math.log(5), math.log(10), math.log(20), math.log(40), math.log(100), math.log(250)], [0, 5, 10, 20, 40, 100, 250]):
		for pmi_type in ['ppmi']:
			for cds in [1.]:  # [1., 0.75]:
				for window_size in [2, 5, 10]:  # [2, 1, 5]:# [5, 2]:
					print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}; ...'.format(
						pmi_type, window_size, cds, sppmi))
					transformed_out_path = os.path.join('/disk/data/tkober/_datasets/1b_word_benchmark/', 'wort_vectors',
														'wort_model_ppmi_lemma-False_pos-False_window-{}_cds-{}-sppmi_shift-{}'.format(
															window_size, cds, sppmi
														))
					if (not os.path.exists(transformed_out_path)):
						cache_path = os.path.join('/disk/data/tkober/_datasets/1b_word_benchmark/', 'wort_cache', )
						if (not os.path.exists(cache_path)):
							os.makedirs(cache_path)

						vec = VSMVectorizer(window_size=window_size, min_frequency=50, cds=cds, weighting=pmi_type,
											word_white_list=whitelist, sppmi_shift=log_sppmi, cache_path=cache_path,
											cache_intermediary_results=True)

						vec.fit(bn_reader)

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


def vectorize_gigaword():
	from discoutils.thesaurus_loader import Vectors
	from wort.datasets import get_miller_charles_30_words
	from wort.datasets import get_rubinstein_goodenough_65_words

	# p = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews_lc_noid_lemma.tsv')
	p = '/disk/data/tkober/_datasets/gigaword/unparsed/gigaword_lc-True_lemma-False.txt'
	giga_reader = TextStreamReader(p)

	# out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors')
	out_path = os.path.join('/disk/data/tkober/_datasets/gigaword/', 'wort_vectors')
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	# whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_mturk_words() | get_men_words() | get_rare_words() | get_simlex_999_words() | get_msr_syntactic_analogies_words() | get_google_analogies_words()
	# whitelist = get_miller_charles_30_words() | get_rubinstein_goodenough_65_words() | get_ws353_words() | get_men_words() | get_simlex_999_words()
	# whitelist = get_ws353_words() | get_ws353_words(similarity_type='similarity') | get_ws353_words(similarity_type='relatedness') | get_men_words() | get_simlex_999_words()
	# Bless words
	#whitelist = 'cloak,screwdriver,spade,corkscrew,car,bed,birch,squirrel,cockroach,bowl,apricot,clarinet,shovel,spinach,bomber,cow,beetle,glider,herring,acacia,pineapple,sofa,whale,cypress,knife,cedar,ant,jet,revolver,corn,deer,fridge,stereo,yacht,horse,wasp,vest,missile,tiger,cat,hornet,donkey,snake,turtle,sweater,cranberry,strawberry,elm,beet,gun,coconut,willow,grape,train,ferry,violin,pine,gorilla,lion,table,poplar,axe,bag,fox,tanker,chisel,hammer,cello,mug,lime,alligator,falcon,crow,dresser,dove,sword,oven,saw,rabbit,elephant,cucumber,carp,cod,dagger,spear,butterfly,robin,coyote,villa,bookcase,freezer,grasshopper,cabbage,scooter,helicopter,goat,flute,truck,lizard,penguin,library,washer,bear,tuna,robe,pigeon,bull,vulture,fighter,oak,castle,owl,sparrow,catfish,parsley,glove,hotel,pig,bottle,rifle,plum,coat,rake,wrench,turnip,television,scarf,grenade,goose,eagle,box,cathedral,cannon,lettuce,rat,couch,jar,toaster,blouse,hawk,broccoli,apple,carrot,frigate,peach,giraffe,celery,potato,pear,wardrobe,cherry,cauliflower,phone,stove,trumpet,hatchet,chair,fork,trout,battleship,desk,piano,woodpecker,saxophone,onion,spoon,bus,mackerel,goldfish,moth,pistol,pheasant,guitar,grapefruit,radish,radio,lemon,sieve,musket,ambulance,van,salmon,banana,garlic,beaver,restaurant,dress,shirt,dishwasher,dolphin,swan,cottage,hospital,pub,sheep,jacket,hat,bomb,frog,motorcycle'.split(
	#	',')
	# ML 2010 words
	# whitelist = ['achieve', 'acquire', 'action', 'activity', 'address', 'age', 'agency', 'air', 'allowance', 'american', 'amount', 'area', 'arm', 'ask', 'assembly', 'assistant', 'attend', 'attention', 'authority', 'basic', 'battle', 'bedroom', 'begin', 'benefit', 'better', 'black', 'board', 'body', 'book', 'building', 'bus', 'business', 'buy', 'call', 'capital', 'care', 'career', 'case', 'cause', 'central', 'centre', 'certain', 'charge', 'child', 'circumstance', 'city', 'close', 'club', 'cold', 'collect', 'college', 'committee', 'community', 'company', 'computer', 'condition', 'conference', 'consider', 'contract', 'control', 'cost', 'council', 'country', 'county', 'course', 'credit', 'cross', 'cut', 'dark', 'datum', 'day', 'defence', 'demand', 'department', 'develop', 'development', 'different', 'difficulty', 'director', 'discuss', 'door', 'drink', 'earlier', 'early', 'economic', 'economy', 'education', 'effect', 'effective', 'efficient', 'elderly', 'emphasise', 'encourage', 'end', 'environment', 'european', 'evening', 'event', 'evidence', 'example', 'exercise', 'express', 'eye', 'face', 'family', 'federal', 'fight', 'follow', 'football', 'form', 'further', 'future', 'game', 'general', 'good', 'government', 'great', 'group', 'hair', 'hall', 'hand', 'head', 'health', 'hear', 'help', 'high', 'hold', 'home', 'hot', 'house', 'housing', 'importance', 'important', 'increase', 'industrial', 'industry', 'influence', 'information', 'injury', 'intelligence', 'interest', 'intervention', 'issue', 'job', 'join', 'kind', 'kitchen', 'knowledge', 'labour', 'lady', 'land', 'language', 'large', 'law', 'leader', 'league', 'leave', 'left', 'letter', 'level', 'life', 'lift', 'like', 'line', 'little', 'local', 'long', 'loss', 'low', 'major', 'majority', 'man', 'management', 'manager', 'market', 'marketing', 'match', 'matter', 'meet', 'meeting', 'member', 'message', 'method', 'minister', 'modern', 'name', 'national', 'need', 'new', 'news', 'northern', 'number', 'offer', 'office', 'officer', 'official', 'oil', 'old', 'older', 'opposition', 'part', 'particular', 'party', 'pass', 'pay', 'people', 'period', 'person', 'personnel', 'phone', 'place', 'plan', 'planning', 'play', 'point', 'policy', 'political', 'pose', 'position', 'pour', 'power', 'practical', 'present', 'previous', 'price', 'principle', 'problem', 'produce', 'programme', 'project', 'property', 'provide', 'public', 'quantity', 'question', 'railway', 'raise', 'rate', 'reach', 'read', 'receive', 'reduce', 'region', 'remember', 'require', 'requirement', 'research', 'result', 'right', 'road', 'role', 'room', 'rule', 'rural', 'satisfy', 'secretary', 'security', 'sell', 'send', 'service', 'set', 'share', 'short', 'shut', 'significant', 'similar', 'situation', 'skill', 'small', 'social', 'special', 'stage', 'start', 'state', 'station', 'stress', 'stretch', 'structure', 'study', 'suffer', 'support', 'system', 'tax', 'tea', 'technique', 'technology', 'telephone', 'television', 'test', 'time', 'town', 'training', 'treatment', 'tv', 'unit', 'use', 'various', 'vast', 'view', 'wage', 'war', 'water', 'wave', 'way', 'weather', 'whole', 'win', 'window', 'woman', 'word', 'work', 'worker', 'world', 'write']

	# Tensed Entailment whitelist
	whitelist = 'knead,fuelled,charging,tear,manipulating,sobering,rending,avoided,partner,feasting,mined,shop,blurred,committed,contrast,agree,exercise,evaluated,poured,coercing,deliberated,jigged,run,healed,conserving,substituted,sloshed,sinking,moored,truncheon,slurping,haying,mounting,hanging,pinked,disagreeing,fence,stutter,plug,pulping,cascaded,print,deleting,persisting,cohering,stigmatize,prick,responding,rest,chagrin,visualize,negotiating,thrived,enliven,bantering,terrifying,schemed,reserve,mention,kicking,enraged,knight,scram,toddle,threw,trap,salivating,hoisting,bounce,unbuckle,manipulate,retract,incite,agglomerated,sifting,federated,buy,clack,astound,commercialized,obscuring,cap,reciting,shingling,dip,wetting,foresaw,save,slobber,nosh,diced,sheltering,jolted,stifled,nettle,reconstitute,hurled,bug,boomed,bisect,dazzle,signalled,hunted,chopped,kill,tingle,object,surge,eyed,prophesying,envisaging,goggle,distorting,termed,preen,quiz,muzzle,boated,startled,curse,sidestep,reeking,wading,escort,isolate,refreshed,skyrocket,uplifting,plopped,levitated,robed,deafening,mangle,paving,prospecting,perfecting,rapped,cawed,gauged,staggering,chastise,unbolted,jumping,transcribe,sunk,warehousing,link,mingled,communicate,hoped,strip,air,perking,triumph,wither,dispirited,gaining,leak,ravaged,snort,surprised,lunch,equip,weeding,grasped,baptize,shorten,diversify,twiddled,diagnosing,uplifted,spared,strapping,belch,waltzing,crippling,slug,checking,cleanse,paining,pose,scheduling,core,scouted,bared,serve,voiding,unlace,protected,enamelled,storming,dislocated,rise,appearing,vacating,sorrowing,ravaging,unleash,deploring,branded,design,rue,enticing,intensified,oystering,disperse,tormenting,gab,picked,salivated,wrestling,settle,heaving,decomposed,cherishing,farting,formed,retracted,waltz,overthrowing,won,stirring,scarred,disconcerting,blacken,strain,teeming,deprecating,cabling,retiring,work,cautioning,crash,hear,discovered,leash,exacted,teased,bluffing,intermingle,ornamenting,befell,linked,dared,combining,protruding,store,improvise,conditioning,anchoring,decline,yearning,warble,inverted,overcooking,straddled,crowding,ruffling,joining,evicting,fried,lengthened,molest,chipping,pitting,sow,pronounce,snooping,cooped,admonish,written,roused,kissing,deprecate,abounded,bifurcating,unfold,added,debark,forward,predicted,peering,faulting,outbidding,stash,bridled,cruised,streamed,bored,grazing,crumbled,complimenting,push,quieting,seethed,hosed,wheedled,reverted,stiffened,trimmed,wresting,alternating,exiled,placed,coached,worrying,flouring,following,scar,congratulating,embittering,perambulate,filed,reeked,hailed,reprove,harming,throbbing,starred,fainted,westernized,unbuttoned,chattering,drawling,demanding,thread,yanked,convict,appeased,swoosh,thumbing,tearing,pioneer,outwitting,waiting,resulting,padlocking,generating,raging,sent,hook,compare,tempting,hiccuped,ransacking,zigzag,claiming,professed,tottered,degrease,lilt,devaluing,buttoning,represented,taunting,overreached,produced,accumulating,prosecuting,sloping,slinking,hitched,endured,booking,quivering,neutralizing,impairing,nip,referred,dislocate,leashing,mobilized,catching,overburdening,sprouting,dye,abridging,boiled,waded,confide,gulled,ladle,reclined,tailored,causing,pique,wilt,ionize,shattering,fretted,console,shifted,combing,smothering,swirled,dotted,plotting,listened,lose,salved,roofing,punching,firm,stole,boggled,agitated,corral,beaming,boss,outweighing,steepen,intrigued,thronged,ship,dyed,nipped,crushing,interleave,acknowledging,chopping,imposed,caused,reproved,grazed,whelp,shrivelled,titillate,disentangling,starching,bracket,delight,tidied,express,depressurizing,surging,hopped,blanket,itching,reanimate,stained,beat,unchained,honeymooned,swathe,cycled,bake,equivocated,gilling,canting,sewing,observed,rusting,chirping,choked,satisfied,drape,blinking,translate,sobered,sugaring,tying,dimmed,trumpeting,clipping,praising,tempted,grasping,value,disengage,sequester,chugged,reelecting,ranking,cash,conjoined,disliked,detecting,matching,sliced,dissolving,tape,oiling,corking,championed,quarter,rip,lolling,contextualizing,sacrificing,lending,aging,outraged,defending,concealed,cube,date,knew,challenged,outweigh,befuddling,sniffled,competing,huddle,napping,bay,despised,professing,stupefied,photographing,direct,muttering,rebel,snapping,surrounding,annotated,injuring,generated,preened,lilting,vexing,declare,hobnobbed,hugging,idolize,expire,figuring,jade,imprisoned,eject,stirred,fluorescing,storm,smoked,gathered,flip,potter,pillory,solidify,removing,fixed,flank,wreck,clink,manacled,occupying,budding,hewed,quarrel,brained,thumping,proffered,impel,blending,crabbed,smothered,squalling,stroke,longing,provoke,disgorging,coalesced,pool,hallucinated,tarmac,studied,stating,fatigue,whistling,brawling,allotting,rattling,eat,trusted,formulate,stitched,muzzling,pester,clasping,exported,disorganizing,prune,rearm,rhapsodize,heard,obsess,clouding,conveying,glorifying,revived,singeing,clenching,socked,thwacked,cajoled,lighten,inserting,bloom,honeymoon,waking,cautioned,gorging,associate,sprained,changed,napped,crediting,amplified,harmonized,intimidate,disassemble,tumble,developed,annexing,cuff,wallpapering,prancing,preach,dried,lambaste,fool,disappeared,regard,endowed,slating,embellished,bashing,maintaining,traverse,forbid,ink,tame,forming,pleasing,tailor,outsmarting,rhapsodized,challenge,steady,consecrating,regain,activated,lounging,veered,dripping,raining,thrashed,prodding,roaring,bewildering,brawl,puking,pierced,shrugged,splintered,surveyed,undertake,soiled,dealing,pirated,grieving,overhearing,nail,shuttering,robing,missing,could,constructed,camp,transcending,convicting,incensing,crimped,overheating,living,discussed,clinked,incense,assuaging,coagulated,wrinkle,crumpling,inverting,slandering,sharpen,amassed,banging,rule,cringed,clunking,amused,appall,disturbed,planting,mentioning,bordered,receding,anger,canvassing,loafing,mothering,waggled,clothing,depict,creak,teeter,plating,blot,dawning,payed,anoint,gazing,huddling,cuffing,stopping,scalping,flicked,eroding,anesthetize,overreach,yawned,starch,forgiving,popularize,forfeit,erased,plague,notice,shrunk,exclaimed,resented,covet,exterminating,close,duped,intruded,sagged,frisk,dally,puzzle,conferred,parading,equipped,terming,blackened,emancipated,overwhelm,dithered,confessing,hastening,purring,bombarded,winking,enthrall,suggest,breastfeed,lisping,scorching,caught,upgrading,appraised,enlighten,festooning,summered,exporting,jogged,spotted,billet,sail,inherit,reopening,inscribe,play,strengthened,totter,rejuvenated,contrasted,kiss,enlist,fermenting,transmitted,tutored,strew,interleaving,overturn,parachute,accelerate,bleached,murdered,strained,ruing,quickstep,squeal,roughened,disfigure,mesmerized,knock,flinch,botch,shower,unfolded,undercharged,dazed,subsisted,dismaying,replenish,awoke,affronting,requiring,separating,asserting,calibrating,rabbit,steam,pin,forge,jingling,thirsting,bandaging,mechanizing,efface,rued,leaked,shunned,burgling,dissolved,qualify,watered,perturb,rutting,blinked,printed,chatted,approve,notifying,garland,solder,assess,camouflaging,braised,thinking,beautify,persecute,catnapped,fluttered,chop,propagated,diverging,plough,unfurled,jingled,sidled,cited,torturing,exalting,hinged,insinuate,controlled,merge,long,steadied,stabilized,blotting,police,dying,prostitute,ornamented,entrance,pardoning,traipsed,prevent,bilked,inquiring,pounded,sparkle,enchant,promulgated,humiliate,reel,extorting,lope,taunted,rambling,subsisting,inked,repose,inspect,splashed,intone,corralling,running,dial,shimmered,ticked,stapling,crack,pave,flirt,encourage,bowling,discharged,intertwined,distress,veiled,fizzle,suck,butting,sneered,frustrated,hunger,organizing,slim,discouraged,worried,sharking,wiggle,writing,caddied,yell,changing,grouping,sickened,gouging,disabuse,adhere,ransacked,boxing,moved,bulge,undulated,gash,introduced,loosing,stultifying,whinny,throb,grow,embarrassing,fume,activate,prospering,swooning,tagged,waddle,jarred,knitting,reviling,cried,bicycling,depriving,plaguing,intoxicated,crossing,jiggle,uniting,prevented,fumbling,manifest,damning,scratching,disorganize,disquiet,sounding,abducted,raged,extruded,jail,whaled,destroy,respected,bundle,incinerated,swab,reserved,teetered,elucidated,stitching,debate,slouch,mounded,transmute,zigzagged,invoicing,examined,choking,deduct,fade,reported,sojourning,partnered,lent,nabbed,squelched,expose,incinerating,extorted,delayed,box,transfix,menaced,commenting,passing,devoured,crumple,brightening,asphalting,dispute,bottled,bewilder,cohered,hired,suffocating,accompany,divesting,accepting,leached,transfixed,sacrificed,arise,sharpening,caddying,court,walk,heated,feeing,coop,recommending,dispatching,whack,chose,circling,panic,row,resumed,banishing,stoop,emerged,rushed,collapse,exhilarate,shut,injecting,round,dispersed,strewed,harvesting,mingle,shattered,alerted,outstripping,clasp,opposed,stew,broadcasted,swindle,washed,twisting,polish,imprinting,congratulated,untying,swooshing,devalued,tinkle,predominated,salvage,bloodied,boycott,nuzzled,mashing,fluttering,defamed,stated,kitten,applauded,meeting,slamming,stowing,caulking,muddy,bicycled,trundled,creep,entranced,puzzling,tanning,subsided,heal,describing,hinge,unnerving,quivered,shear,registering,proceeded,swipe,splayed,superimposed,flopping,disabusing,wrecked,wandered,withdraw,splash,powdering,omitting,telex,remit,tanned,exceeded,radiated,waned,meditating,require,depressed,officiated,repossessing,brought,shiver,varnishing,sorrow,desist,unlatch,retching,admiring,lasso,elucidating,hybridizing,leaf,lob,deform,keeping,sliming,photocopying,burl,invigorate,sprout,twitched,snack,democratizing,scramble,posting,refrain,niggling,bivouac,submit,hooking,bugging,vacationed,coupled,summer,jogging,pasting,stowed,reflect,astounding,envisioning,tinting,demand,overthrown,emptying,topping,chuckling,gilding,tainted,plucking,tensed,larding,lounged,clanging,punt,sobbed,walloped,lowered,closing,face,treating,dismounted,decimate,rebuking,slit,unfix,reopen,bristle,eradicating,explained,wallowing,preparing,sponsor,animate,chirruping,regret,succeeding,ban,galvanized,overhanging,weeping,pillorying,burping,sweltered,patter,wrenched,adopting,animating,unionize,widow,grooming,wanting,sing,ushered,denude,grinning,emblazoning,plait,writhe,dove,oozed,stabilizing,pasteurizing,bucking,felt,glimpsed,tamped,chased,attain,append,coalescing,idolizing,blanketed,total,meandered,bewailed,anchor,skate,leavening,telephoned,dubbing,feuded,umpire,decking,decide,skirmished,nibbling,compliment,seducing,hissed,explicate,potting,burbling,poaching,dab,pricing,degreased,parting,gobbled,strike,gauge,sock,outlaw,compromising,puffing,bask,decay,tumbling,locking,lace,caress,shuttled,persevered,sterilize,chaperoned,cruising,blew,lament,deserting,hiring,arriving,recognizing,debilitated,shadowing,disrobe,rile,wetted,fix,reappear,propagating,snip,choreograph,denigrating,admonished,trekked,remember,encompass,depress,pealing,sopping,desecrated,magnifying,peeve,glower,parched,relished,dulled,extracted,happened,grate,lecture,shake,decry,skirted,stretch,cascading,enter,initialling,moult,whimper,attaching,strangling,afflict,designating,crate,file,mourning,drug,concur,disbelieving,resounding,hunch,improving,discriminated,licked,chimed,claimed,permitted,pooled,resigning,calcify,fining,peeking,sidle,leashed,driving,saddle,implied,flushing,gratify,evacuate,reproving,interspersed,overlapping,congregate,inflamed,beaten,ventilating,perused,capsizing,suppose,insinuating,dictated,condemning,void,exposing,chitchatting,evict,redeeming,exhilarating,discriminating,char,manifested,elaborating,perspire,sand,conceding,plop,quaffing,cowing,scrub,rafting,eased,construct,pointing,smile,thinned,crease,illuminating,gamble,consorted,gambol,squeezing,untied,executing,dismay,flame,fitting,urinating,abolished,span,gushing,paved,classed,poked,smother,yelp,abstracting,weekending,chasten,cheat,simmering,confounded,scorning,raced,bottle,exhaling,transmuting,spilt,howling,stimulate,shrilled,carving,unsettle,crossed,pawed,sit,loaded,coiling,repeating,chauffeuring,holiday,speed,talc,baited,carry,floored,guide,stain,ceding,reside,yowl,giving,zest,investigate,blanching,planning,pricked,varying,tamed,scheme,hoodwinking,talking,addressing,obliterated,staffed,hijacked,bussing,blabbing,reduce,limping,initiate,cursing,recover,jutted,fetched,enlisting,encouraged,containing,hypnotizing,coughed,sketch,gurgling,exploded,scalped,glorify,quarantined,nudge,embolden,breaking,chance,peg,maddening,disheartening,gloated,flanked,coach,secure,opened,raze,dedicated,swirling,loaning,obtained,elated,goggled,cite,hover,wandering,excuse,denigrated,confirming,languish,disliking,scavenged,manicuring,scraping,distemper,elaborated,dredged,masking,scent,slopping,scratch,reposed,divorcing,agonizing,pod,maligned,scorch,defrosted,rearrange,ramming,fettered,humidifying,quibbled,ordering,grousing,networking,mailing,huddled,plodded,wincing,boning,expand,squared,impregnate,basked,stem,titter,splaying,entrapped,whittling,repelling,move,exasperating,trolled,riveted,inundating,pound,puckered,severing,shaking,studding,bellowed,guffawed,emblazon,shopping,conquering,interest,incised,frosting,strut,recurring,entangling,objectified,costing,angering,flopped,plot,order,live,appeared,enduring,invested,ascend,shovelled,anesthetizing,blushing,neglect,tie,suppressed,veneered,startling,clobber,swept,filming,drooping,pegged,assemble,rankled,bashed,loosened,grating,stinging,treat,unionized,offer,blessing,glaze,revolt,robbed,furnish,liquefied,pruned,crisping,cherished,hooked,sloped,spray,credit,shadow,dominating,slogged,mizzle,pilfered,distracting,laud,squint,trooping,record,lighted,perforate,persecuting,reproach,gave,crisscross,count,estimating,starting,speaking,creating,immobilizing,wheel,breathed,brayed,nettling,ticket,berth,clamped,flung,hocking,catapulted,disapproved,yearned,bend,beginning,procure,glide,scooping,chink,separate,stomped,chastised,quadrupled,delving,impregnated,burble,warn,patent,plundered,pressured,hosted,whirling,loll,trundle,christening,sob,bop,coughing,tote,infecting,splashing,clapping,deflowering,chewing,buck,eloping,mollifying,bracketing,handled,smuggle,caring,conflicted,deem,assembling,relaxed,stroked,cleansing,expectorated,reinstating,interlinking,enlisted,droning,detail,rationing,remain,sleepwalk,drilled,missed,quieted,cook,biting,skied,adjoined,macerated,fettering,martyred,reviled,persuade,flaunted,repudiate,dazzling,unfurl,razed,measuring,meandering,repelled,rattled,shove,westernizing,lured,confuse,gutted,maintained,grilling,exceeding,planking,twitch,lend,illuminate,mourn,triumphed,classing,bowed,moderated,insist,projected,roiling,traipse,thwack,connected,chronicling,flooding,teetering,gleaming,dazzled,lambasting,tingled,writhed,pardon,lectured,mowed,deprive,floated,divide,witnessing,divest,narrate,telexed,pipping,established,clomped,browned,ascended,dehydrating,swaying,evaluate,rob,protrude,squinting,initiated,notching,dissociating,unhinge,witnessed,distrusted,normalize,published,scream,clobbering,swat,relaying,conspired,comment,saturate,wove,inflating,shutting,reanimated,commingled,extradited,triple,inhaled,dub,acquiring,canonizing,skirting,superimpose,cloaking,bombed,tarnish,explicated,bar,pulsating,arousing,inking,falter,staining,modulated,fuelling,intermix,outrage,raising,affecting,burling,jailed,owning,rewarded,tether,dangled,injured,choosing,knotted,compress,lure,quartering,walloping,electrified,disagree,deter,emulsify,cheapen,viewing,snatched,flinched,proffer,acquitted,dabbed,alarm,diversified,happening,banter,sputtered,banded,tuck,pen,produce,kneel,conceal,intertwining,showing,rose,like,weighted,blister,networked,report,thatching,fazed,calming,team,remained,clearing,turfing,patting,cool,assumed,criticize,modelled,dissipating,enrage,crunched,outsmart,propagate,blend,masqueraded,mutilated,clank,paying,ending,shoulder,fill,dislodging,entered,perturbed,hardening,christen,pencilling,bought,ruled,blinded,coxed,augment,clouded,belted,predominate,freezing,elect,discombobulate,coasting,fearing,misled,crawling,pray,clustered,saw,doused,recalled,hose,embraced,genuflecting,hypnotize,hijacking,restraining,restoring,perfect,star,shuttling,slobbering,defining,encircle,retaliate,strutted,garnished,unzipped,ameliorated,transformed,subsiding,bleach,strutting,gravel,enquire,supporting,conga,implying,jousting,situating,touching,pioneering,seethe,tangoing,deforested,alert,purging,sigh,drizzling,draining,roaming,flaunting,swoon,diagnose,shampooing,warming,searching,click,prod,united,enroll,bait,charge,composing,flanking,resent,trilling,left,eulogized,ski,detected,drifting,insinuated,secluding,offend,qualifying,bleating,convene,murder,fog,embarrassed,spur,fabricating,nudged,tiptoed,decorating,respect,address,clutched,feminized,officiating,sought,devastating,bristling,inspecting,holding,kidnap,levelling,can,teething,pressing,frothed,plundering,eddying,braise,babbled,collect,retreat,cauterized,predict,ate,bugged,stump,crocheting,slinging,splotch,exhausting,shellacking,corresponding,dwell,elevated,fawning,spare,hankered,swallow,loaf,ululating,confirmed,uplift,reprimanded,boggle,milking,jetting,deluge,tinned,unify,house,deposed,turf,explored,fumed,swerved,spooned,chain,decree,settling,mesmerizing,aired,designing,pacified,inspire,exploding,flinching,imploring,crayoned,rotating,meaning,attire,mollified,vacate,visit,hold,rouging,quaking,dash,heeling,degenerated,cadged,harvested,burn,lease,chomping,handle,hugged,draped,plying,condense,disparage,watering,dispossessed,demolish,expiring,matured,crept,bilking,glanced,skittered,dance,schedule,circle,concern,toss,peeked,scale,mused,blink,rejuvenate,characterize,ionized,quietened,scoffing,westernize,welding,caravan,taxied,wedged,hindering,felicitated,celebrating,fleecing,bless,respiring,collated,guarded,peered,understood,dialling,wiped,craning,ran,qualified,squaring,exercised,aggravate,liked,stiffen,copping,asphalt,pull,deforest,enrapturing,civilizing,deify,spice,tugging,straying,feast,rush,asphyxiating,annihilating,enthralled,quarried,shuddered,positioning,discontinue,stealing,investigated,tally,recognize,jumbled,glow,parade,clobbered,miss,forbidding,unlock,scavenge,volunteer,vitrifying,assassinate,commended,airmail,pumping,pursed,harried,cauterizing,combat,contextualize,stink,twiddle,macerating,slope,wreathing,discerned,bewailing,convicted,unleashing,interviewed,trucked,refraining,undressing,transporting,institutionalizing,unwinding,exalted,gouge,diverted,honoring,heap,contribute,permit,crisp,commit,cheating,burgeoning,discomfited,pestered,bargaining,starved,hoe,invigorated,desiccating,surviving,pursing,overthrow,clutch,douse,burbled,sullied,casing,loop,poach,vilified,approach,chlorinated,denouncing,dispatched,strayed,mumbling,drawl,donating,indicated,schmooze,stimulated,coin,emerge,penalized,frame,ticketed,compiling,congregating,desire,birch,reimbursing,deforesting,flamed,announced,burled,snoozed,curved,taxing,deliberating,enthuse,rowed,awarded,endow,cable,slosh,charged,donned,brushing,thudding,gagged,slowed,cashed,banqueting,waver,heightened,incorporating,murmur,shorting,stumble,despoiled,nodded,announce,accumulated,twist,quacking,beguiled,shivering,grumbled,mass,peppering,captained,knifed,zooming,herding,fatiguing,singe,matter,prohibiting,bleeding,firmed,slog,cauterize,canoe,suspended,chase,diffused,appealing,flourishing,sealing,baring,trim,disfigured,omit,hit,claw,hurl,anglicize,abridge,curve,affected,defend,lubricate,woofed,moving,procured,flapped,roofed,complimented,adorn,saddening,restricting,awarding,concede,stagnating,coerce,lumbered,finding,respond,clutter,crackled,depreciated,penning,oust,loosen,titillating,erupt,stroking,enraging,concentrate,win,oscillate,interlaced,connecting,measure,escorted,interchanging,ply,abolish,dissuade,vacuum,mortifying,plumbing,ravish,repulsing,illuminated,veil,articulate,blurt,flowering,glided,photograph,rocket,coat,impeded,photocopied,reheat,spot,gasp,jangle,shortened,clean,crown,pearl,incinerate,awoken,shade,desisted,overcook,spewed,whistled,appropriating,clash,hefted,rumble,computed,weaning,harmonizing,fluster,humiliated,float,deplore,sauntered,vomit,emerging,revering,pestering,loped,supported,battering,woof,showered,cleared,trot,appended,fatten,sounded,puttying,swaggering,install,warping,cremating,inspected,doped,limp,cuddling,impressing,paddle,adored,guaranteed,slackened,degraded,flaring,sever,tack,clerked,angered,associated,spelt,track,traded,damming,bailed,gape,lamenting,duelled,revolve,consent,correlated,ridded,quarantine,hope,warning,speckled,declaring,scribbling,filleted,watch,rowing,peal,conducted,unzip,crashed,germinated,came,seating,exhibiting,awake,designed,crated,guess,enforce,emboldened,gabbing,joined,clomping,crippled,posing,informing,reprimanding,salt,fidgeting,explode,enhance,intoxicating,ogling,spanned,interacting,equalized,flushed,quiet,doubling,clubbed,exulted,bribed,interchange,panelling,devoted,tasting,groomed,executed,informed,polarize,reposing,slink,hindered,wangled,derived,asserted,abused,sojourn,toil,harassed,clutching,manifesting,recruiting,discriminate,corrupting,marvel,dithering,flirting,filling,bullying,rage,trembling,fascinated,scudded,hallucinating,reclaiming,framing,disquieting,butcher,skipper,mixed,wrangled,restricted,spouting,skedaddled,nicknamed,scrape,cementing,attempting,microfilming,nosing,prodded,threaten,attenuate,rally,shaming,spawn,engrossed,roil,stooped,prevailed,lifting,whitewashing,gloved,label,love,telegraphed,rearming,anesthetized,indicting,oxidizing,baptizing,retched,championing,aim,respecting,wakening,wreathe,sniffling,dusted,securing,droned,cajole,demoralizing,corked,tremble,emboldening,quadrupling,excommunicated,theorize,embroider,insisted,shrilling,squished,exterminated,prowling,grin,espied,included,thought,pinion,interpreting,overran,bruising,silvering,implore,colluded,degreasing,dissociated,gyrated,seizing,rutted,flogging,punished,protecting,control,compile,massing,craved,edit,exhaust,wad,warm,inserted,adhering,broadened,disgusted,reassured,twinkling,emboss,yachting,winked,miffed,sheathed,eying,bombing,squeaked,defrost,muzzled,fastened,trickle,leading,resort,shampoo,padlocked,enacted,despair,assisted,departed,breastfeeding,buttonholing,unrolling,buckled,handed,shame,garrotting,mean,smear,hammered,soling,liquefy,garage,glimmering,allure,shingle,aided,hesitated,pout,ooze,burped,littered,obscure,caw,paging,fidgeted,panelled,caramelize,groom,glorying,disintegrating,jig,elected,facing,disembarked,boggling,loose,garrotte,strolled,volunteered,gaze,rotate,squawk,blacking,fizzed,wind,fertilized,liquefying,separated,producing,foretelling,supplied,awing,pucker,choke,hinted,bamboozling,souring,stacking,insuring,erupted,decimating,swear,expunged,reverberate,recollect,owed,decelerating,sole,dispossessing,nursing,debilitating,tangoed,implored,terrorizing,acquitting,sprucing,hurtling,flowed,allowing,instruct,bandage,shaped,enjoy,emanating,deicing,growing,sandpapering,cloak,prowled,plod,harden,leant,thanking,argue,japanned,smoking,scrounging,plucked,slaying,dancing,wrangle,undid,excel,rammed,preoccupy,shrimp,lofting,thawing,expel,bivouacking,faulted,impaired,skulking,blaspheme,spout,slated,stoning,support,appeasing,imprinted,freeze,conduct,worsening,melt,associating,gashing,smelt,risk,whacked,consented,tarnished,disassociated,rustled,terminate,expressed,dream,disclose,ripping,corrupted,content,alarming,floured,mark,kneaded,snivel,mounted,exonerate,stare,desalted,persist,materialize,sweetened,cutting,scolding,desecrate,breathe,committing,chilling,rub,squirt,conciliating,prattling,replaced,hijack,seeding,overstepping,recovering,waving,bandaged,quenched,impelled,phoning,mourned,winnowed,radioed,catch,declined,spattered,ballooned,exhausted,calved,compromise,walking,issuing,charred,invent,rung,caution,curl,shutter,cooed,agitating,blended,nailing,directed,convincing,diversifying,searched,dig,mutate,squirming,cropping,expanded,tiptoeing,dissolve,erase,bewitch,jerk,honor,taper,rubbed,chartering,pedaling,daunting,bearding,fastening,harmonize,undressed,electrocuted,rosin,roasted,publicize,spread,dishearten,tapping,repute,lather,kindle,bound,disburse,mocking,swooned,foaling,unhooking,yoked,envied,skinned,germinate,satisfying,confound,confusing,aching,swagger,snoring,conquered,assist,unbolt,decomposing,deducted,anglicized,catnapping,vying,tin,hobbling,dividing,weed,mope,clasped,exonerating,jamming,blasted,crop,prying,allot,whisk,engraved,perished,curtaining,impress,creaking,stocked,chasing,massacre,lampooning,owing,hungered,sneaking,referring,fashioned,incubated,unwound,gagging,glowering,saluting,clanged,bent,taught,snipped,craving,attained,afflicting,climbed,tweeted,enrolling,canted,assure,reappeared,autographed,posed,bare,heating,suffering,desiring,jiggled,disbelieve,copped,orphaned,persecuted,scuffling,communicated,sipping,loft,confounding,followed,slugging,foam,speared,blossomed,dump,expectorate,offending,shuck,scamper,smeared,shine,infuriated,trapped,buffet,bestriding,peruse,remunerate,theorized,glinted,evening,scouring,rambled,provoked,worked,diminishing,blab,taxi,capsized,checked,zipped,blossoming,binding,plunking,promulgate,surmounted,introducing,mock,agonized,canvass,slide,hobbled,blitzing,inject,pawned,energized,spank,chitchat,convulsing,tootle,denuding,zipping,model,bricked,hoed,institutionalize,schlep,paged,answering,unburdened,deluging,yapping,turned,pocket,toured,quickened,guzzling,sputtering,captured,conquer,insure,liquidating,squinted,engraving,categorized,steamed,swilled,coexist,galling,swapping,stencil,distinguished,dine,stud,reeling,shedding,hoist,supposed,desinged,write,convulsed,breakfast,covering,extort,enriched,curried,stub,aggregated,patrolled,crushed,sponsored,shelling,herd,stamp,casting,slithering,vaulting,intersected,wanted,slither,stopped,snarl,correlate,appointing,hurry,stooping,clenched,shunted,eating,licking,towel,improvised,paddling,promising,whistle,vibrate,speckle,sensing,spitting,wobble,banged,papering,confer,fading,sighted,scolded,cloaked,demagnetized,poking,clomp,censured,carrying,assembled,trundling,wolf,assail,confessed,camouflage,perfuming,smashing,pulling,bench,varnished,helped,intermingled,conveyed,mowing,distressing,prance,funnel,diminished,cawing,rebelling,battling,functioned,steered,exiling,sugar,remunerated,blanch,castigated,rallying,tattooing,experience,undertook,leased,charring,flashing,thudded,sighting,sprain,detonate,broiled,stubbing,narrated,carbonized,page,paste,tunnelled,bribing,deluded,feathering,raft,don,contriving,littering,enhancing,asked,guarantee,hailing,insert,muse,wallpapered,wheeling,snapped,consuming,skewed,plotted,evading,soak,conked,despising,pass,knelt,normalized,conjectured,tick,stashed,zinged,cooperated,enriching,transcribing,cheered,growl,awe,whirled,uproot,raiding,dined,revolving,dealt,tripled,convert,misspelling,emasculated,zinging,electing,arguing,sully,decorate,winced,closed,weaned,emanate,scattering,hankering,brewed,fine,mate,abutting,winkled,adore,met,tagging,heightening,assassinated,devoting,adjoin,peddle,bluffed,incise,excising,invoice,defaming,sicken,secularize,creasing,clapped,assume,increase,referee,squealed,decrease,emancipating,disengaged,lilted,tax,colluding,annotating,sloshing,piping,intended,persuading,dampen,tootling,echoing,wiggling,stupefy,drummed,constructing,cheapened,intoning,sledge,staging,walked,limited,plant,assault,harrying,tootled,doodle,bucked,confided,slammed,hating,trick,spell,hinging,bunted,sandpapered,barring,churn,mottle,counted,harness,relieved,tow,denigrate,rummage,easing,desert,integrating,lied,dong,ripped,groaning,pulped,surmise,noshed,appalling,anguished,thrummed,seek,experienced,recovered,screwed,breastfed,remitted,punted,vacillating,confiscating,battled,salve,obscured,roosted,pulsate,mailed,nabbing,analyze,assuming,patched,brew,tour,belting,cease,shucking,soaked,ease,promenaded,tightened,gazed,nutted,sweeten,inspiring,overlap,travelled,convening,banning,burgle,thrum,denting,betting,surrendered,galled,smashed,copied,arrange,incited,drying,delighted,tarried,surmount,required,crowd,baffling,discomfiting,gratifying,ferret,gawk,banished,block,wrestle,overcame,discombobulated,pomaded,interleaved,tolled,sip,enthralling,blamed,prosecuted,stemmed,excommunicating,articulated,convinced,studded,tugged,masquerading,ambling,spoiling,aggravating,curving,consorting,stood,wail,attempted,cajoling,allocate,borrow,intend,vilify,designated,umpired,spurred,freeing,leavened,cycle,obsessed,apportioning,abrade,evacuating,chatter,recollecting,haggling,reaffirm,buffeting,unrolled,tucked,establishing,clove,persevering,pasture,recalling,memorize,commingle,disabused,honeymooning,bisected,smudging,boxed,splutter,sapping,liquidate,swearing,lounge,complaining,gladdening,ingesting,judged,piquing,humidify,surrounded,dipping,squirmed,return,consulted,trimming,muddied,shooing,puttered,scheduled,abraded,whinnied,wished,darkened,pipe,abated,lisp,certified,interviewing,seized,tasted,bobbed,sell,studying,formulated,organized,proceed,hand,uttering,swish,coping,train,igniting,bail,sold,arresting,preceding,pulverizing,chugging,itch,nauseated,dismayed,strapped,fogging,outraging,abating,circumvented,crucified,improved,exclaiming,tolerate,giggle,pouted,pining,blast,bully,disclosing,soil,paled,accelerating,burrow,complained,flower,immersing,tossed,coddle,appraise,result,quieten,embarrass,recoiling,coasted,touched,intimate,barred,loafed,fermented,bricking,saturated,batted,mincing,mewing,concoct,screeched,permitting,daze,clay,wriggle,uttered,leap,sunbathing,gulped,distempered,scheming,suspecting,succeed,rotated,intrude,priced,colliding,adjoining,purloined,dawn,alerting,groping,coring,film,abusing,papered,powdered,liberated,identify,grunted,compared,extended,quack,project,promoted,happen,brief,slew,sew,stifle,clucked,shelved,battle,dicing,credited,milk,rate,repressing,veneer,revolted,swaggered,jesting,bark,fetter,foretold,frowned,wiring,amazed,scrawl,immunizing,tired,reorganizing,caressed,hum,twining,deafened,disappearing,invented,shunt,excise,unfasten,expounding,humming,clammed,gabbed,analyzed,pocketed,hobble,foretell,inveigling,billed,martyr,impeached,nicking,recasting,joke,exceed,venerate,crochet,acquired,vanish,shrouded,stumbled,skirmishing,bothered,electrify,granulated,commanded,jarring,garbed,slashed,encircled,charm,suggesting,saddened,cased,infuriating,liberate,abducting,inciting,lampooned,toasted,scared,sizzled,doubting,starched,saying,skating,martyring,pulverized,prosecute,minting,splatter,relishing,dilute,sneezed,rid,padded,amass,bed,whisper,accumulate,furnished,displeased,bludgeoning,slurp,bracketed,cellaring,munched,spicing,quiver,ejected,marinated,alternate,waning,pencil,pad,soothed,spindle,outwit,shrill,hosing,oscillating,esteem,storing,reelect,doctoring,drum,reduced,bronzed,coddled,annoy,cry,reeled,niggled,faltering,chattered,chiding,baking,roared,lightened,quenching,freed,emasculating,beam,disseminate,quartered,escorting,prostituted,bearded,spear,focussing,conversed,accrued,snooped,detailing,beautifying,lettered,treated,inform,envisaged,spar,moderate,waggling,fluctuated,yawning,tease,fatigued,crooned,marinate,seething,harnessed,slack,parching,limit,plastering,withdrawing,tantalized,diminish,lacquering,jabbed,entangle,induced,lisped,censuring,note,piled,buckle,depend,ticking,simmered,disintegrate,trade,forested,arching,contracted,curled,abbreviate,heading,swatted,contrive,masticate,sipped,larded,reproaching,wedge,craned,amusing,shouting,baiting,pale,inflate,willed,eluding,masticating,overstep,beating,exulting,putting,botching,corroding,elucidate,aiding,fold,arming,bunting,teethe,butter,clanking,misting,shrug,flabbergast,open,battered,identified,wilting,publishing,basking,affirm,moisten,invoiced,ravage,overturning,scalp,peck,evaporate,chirp,screech,spook,detailed,retrieved,administering,scalloped,shrieked,state,latch,spurted,blinding,affronted,watched,swig,scrimping,wrinkling,corrode,revolved,topple,civilized,transmit,ride,darken,stabbing,harm,togged,replenished,delaying,mound,shrimping,panicking,seep,urged,garnishing,whipping,incorporated,lacquered,retaliated,draw,salving,berthing,dispirit,deepening,tempt,exile,plate,teamed,abasing,yacht,uprooted,coating,remaining,clogged,glue,remarked,wagering,attack,preening,staffing,divorced,kissed,roost,compelled,sop,elating,tip,mantle,formulating,repossessed,motoring,commend,lulling,scrambled,muddling,radiating,earned,withdrew,eschew,perfected,tan,erode,crowed,packed,castigating,gasped,went,burying,polymerize,entering,borrowing,shining,ruminating,transfixing,agreeing,tamp,articulating,tattooed,see,shipped,deviating,eddied,prayed,garb,overwhelming,puttering,throng,loved,case,ramble,thumb,partnering,criticized,foal,meet,scrutinizing,deterring,memorizing,dehumidifying,moped,acidified,claim,ranged,jibing,broke,buttonhole,delay,sway,revere,deciding,focussed,demobilizing,softening,transmitting,pastured,labouring,scare,modernized,spying,straggling,scoff,cabled,rounding,conferring,assert,shovelling,forwarded,appointed,overcharging,arm,dredging,galvanizing,consolidated,weighting,concerned,consumed,monitoring,asking,deviated,repossess,gnaw,summering,benched,ploughed,trek,holidayed,moralized,emasculate,publicized,crocheted,roast,gathering,outlawed,curry,squish,spattering,rifled,enquiring,stinking,grunt,dodge,bin,skimmed,stammered,deposit,name,stucco,contort,sighed,hissing,board,thrusting,sunbathe,lessened,situate,accelerated,told,crumbling,swung,intending,shucked,pecked,disagreed,interact,sidestepping,nickname,sequestering,tickle,typing,squall,objectifying,depicted,lumber,sidestepped,carpet,twittering,tinkled,hazard,macerate,braying,combating,skirt,calibrated,fashion,dismembering,hanker,persuaded,vaporizing,diverged,swopped,absolving,obligated,whitened,mash,dousing,passed,yelling,trickling,veined,amaze,tarry,nursed,render,ionizing,reap,threaded,blitz,petrify,cage,repeated,tell,jigging,choreographed,slandered,spoke,execute,blush,drip,festoon,tweeting,beget,haggle,sprawled,entrapping,stunned,beeped,alienating,dock,spatter,ululate,wheeled,embroidered,stashing,eventuate,degrade,considering,chanced,confederated,staple,unite,vary,lasted,indicating,including,unbutton,befall,patrol,dart,pleased,remembered,shimmering,figure,condoned,neigh,entrancing,grabbing,dictating,blatted,knotting,spruce,crowded,develop,combined,sleepwalking,banned,proposed,antagonize,twitter,distort,exit,minted,lie,donning,purpose,printing,compose,stuttered,skin,castrating,returning,replying,bubbling,increasing,lard,tantalize,glaring,deterred,picking,knighted,espy,cooked,oxidized,scrubbed,forgave,resume,barter,coerced,estimated,grinned,stagnate,lopped,strengthen,calcified,nicknaming,snooze,spanked,squatted,refunded,deleted,plied,pounding,hacking,consoling,satiating,scampering,leed,term,unlatching,travel,embossed,chrome,immersed,protesting,parachuted,insult,electrocuting,planked,chaining,jibed,applaud,dissent,silvered,spluttered,steeping,ding,extirpated,squawked,foaled,wiggled,democratized,decompose,contaminate,scowling,deliberate,derive,riding,motivating,hurting,ogled,barging,bothering,totalled,materializing,thatch,perturbing,bludgeoned,dotting,widening,protect,varnish,transforming,swigged,impair,enlivened,pop,glowing,carolling,flustering,wilted,signing,hoodwinked,sting,pomade,firing,buffeted,whittle,conjoin,stunning,ridiculed,flatter,excoriated,abhorring,slumped,obtain,salute,reek,carried,blackmail,ameliorating,breaded,yowled,boycotted,belt,fizzling,plan,spraying,jutting,travelling,bud,nipping,circumcised,mottling,dehumidify,glinting,siring,froth,punch,squeaking,decorated,dammed,fossilizing,strolling,deceived,adding,flinging,rotting,reserving,heaped,lapped,consider,placated,sighing,vulcanized,chronicled,lodging,fizz,approving,coaching,treasuring,raised,naturalize,gnashed,cough,tacked,ridiculing,staggered,certifying,waltzed,differentiated,fumble,shave,maturing,mottled,straggled,fart,allocating,gumming,scold,fancy,butted,exonerated,cleaving,disgorge,dropping,bewitching,wowed,discourage,behave,daring,thrilling,consoled,explaining,petrified,buried,training,padlock,saddled,etch,stabled,dwelt,docking,lightening,dissuading,pay,hush,prosper,buzzing,probing,specified,reversing,enervated,promised,unburden,dimming,soled,managed,pirouetted,disdained,rolled,wadded,molt,embitter,conversing,eluded,burgeon,hunched,gutting,repudiated,reuniting,peppered,nauseating,snaring,rocking,anointing,bragging,harass,immolate,precipitated,tumbled,paint,swell,prevail,hovering,fractured,deepened,fracturing,rasp,tossing,obligating,wedging,torment,blackmailed,precede,desalt,cleansed,naming,inducing,spew,envying,roof,lingered,glittering,wheezing,flashed,enlightened,mentioned,contaminating,wielded,champion,robe,rinsing,reprimand,grooved,bivouacked,lifted,playing,sundering,going,sleighing,croaking,suffocate,expunge,forged,reconstituting,pulsated,include,bartered,admonishing,reverberating,hollered,bump,remonstrated,brawled,demanded,swerve,sailed,dehydrate,discussing,refund,burrowing,resounded,wireless,wired,prize,attacking,crimping,quoting,croak,immunize,contrived,guard,commissioning,drowsing,graduated,sung,abstain,buttering,surround,pine,excite,rescue,stomp,flooded,snorted,enjoying,slumbering,equalize,interpreted,stabling,adorning,care,humidified,purchasing,squash,denied,stapled,diving,reassure,reacted,killed,gummed,seduce,imposing,leave,rolling,imagining,humble,troop,clunked,skiing,clamp,collating,noting,argued,defended,lashed,skated,ruminated,buckling,buzz,port,linger,treeing,blared,mocked,relinquished,pouting,unscrewing,ossified,altered,heave,fenced,apostatize,shout,duck,snuff,quitted,undercharge,envisioned,garnish,bubbled,boogie,trust,ululated,narrowing,lunching,flipping,appealed,joust,rebelled,administered,mine,borrowed,settled,nickel,whining,squabble,hiding,aiming,representing,chiselled,harnessing,mulched,retreated,shoot,awaking,debug,reassuring,varied,tile,chartered,bear,menacing,embalm,glowed,treble,outsmarted,trucking,jolting,booming,strewing,swap,cost,washing,reputed,darting,foresee,botched,prepared,unfurling,snickered,reelected,flatting,ring,treasured,wowing,snared,released,bomb,voicing,smiled,scrabbling,fondle,vanishing,broadcasting,scour,jested,shouldered,drained,remitting,enslave,rankling,feather,deceiving,accuse,whine,disseminating,balanced,saved,scrabble,moaning,pummel,strap,part,categorizing,threading,distill,diapered,exude,snickering,imitated,tarred,excited,bellowing,delete,oxidize,emptied,introduce,inflaming,looking,piped,wrangling,activating,damaged,punishing,commenced,gurgled,scoffed,degas,peel,destroyed,needed,fattened,emulsified,jabbered,shortening,abhor,dated,parboiling,sickening,ravished,rusted,twinkled,polishing,removed,shaded,ascertain,telephoning,minded,crashing,rubbing,revitalized,contract,dipped,perplexed,loathe,rewarding,stitch,muting,affiliating,buttonholed,jeer,imprint,scowl,recede,yellowed,rave,discarding,flicker,snipe,overrun,radio,sharpened,donate,hacked,victimizing,stuccoed,mixing,wiping,acidify,restore,stormed,disassembling,camped,completed,recreated,chew,unhook,warp,sandpaper,funnelled,perishing,plagiarizing,thrilled,boring,beach,diapering,felicitating,atrophied,clothed,multiplying,affront,exchange,popping,hollering,ascending,fetching,contextualized,forced,gaped,canonize,smash,substitute,impeach,shaving,scrubbing,dislocating,refresh,unbuckled,partition,crumble,returned,collide,thumped,roughening,reclaim,hiss,rekindled,screened,bloody,dribbling,talk,shock,inebriated,competed,accompanying,enraptured,collecting,clanked,add,pasteurize,alienated,brighten,slipped,berrying,democratize,diffusing,disturb,quench,roam,cosh,welcome,entertained,doodling,grab,rejoice,rebuked,subsist,releasing,straining,rearmed,jangling,thrill,sucking,descending,scraped,corresponded,rouse,writhing,intermingling,chastening,leaching,purge,neck,blasphemed,preoccupied,recruit,umpiring,corrupt,landing,exchanged,grounded,smell,blare,reanimating,congregated,visiting,irk,patch,behaved,chirrup,erasing,wadding,strangled,conjuring,wax,tethered,flecking,wallop,voided,steepened,tango,appease,unfixed,venerated,sculpting,swishing,misspelt,coil,nab,sidling,crunching,invade,bobbing,fossilize,jingle,prop,jangled,paralyze,flow,ferreted,splicing,fabricate,wallow,condemn,thunder,marry,embezzling,even,reddening,avoid,vault,filmed,pointed,fashioning,taped,relay,arose,lessen,led,exiting,oared,tapered,amazing,sowing,raving,guessed,bickered,beeping,singed,pinned,caramelizing,network,sponged,bridged,blanketing,branding,captivated,supplying,clang,pruning,suffused,pricking,winnow,spanking,guffaw,ruffle,lead,expound,demobilized,shovel,scowled,abound,lived,chucked,revitalize,teasing,unburdening,depleting,sweeping,tinkling,chill,is,dare,partitioned,annoying,offended,laced,line,sunder,starring,maddened,het,handling,form,specifying,blindfold,dispiriting,stalling,selecting,lolled,tunnel,grilled,blazed,exasperated,underlie,stage,puff,devote,rumpling,outnumber,enlivening,publish,whispered,kicked,cubed,clump,tormented,deflating,fray,shambling,sleigh,madden,ration,maim,effacing,picturing,abetting,patching,boil,signalling,crush,evaporated,enervate,enrapture,sift,dispatch,severed,depositing,ordained,fell,mortified,partitioning,cringing,drooped,growling,duel,yielded,leasing,annihilated,release,enchanting,sputter,flood,japan,dither,doff,spawned,warehoused,hail,catapulting,excavate,grumble,swooshed,protested,immolating,swatting,rifling,riveting,arranged,flattened,batter,traversing,cede,annotate,worm,asphalted,coarsening,babbling,chained,tucking,enlarging,participated,exterminate,ossify,demagnetize,remarking,ported,mutating,hide,oiled,drawing,multiply,drumming,crystallize,yellow,sprinkling,crewed,mill,welcomed,howl,jaded,refunding,cackled,branch,divert,demagnetizing,shearing,whitening,abut,respodning,erected,relying,motivated,puckering,rushing,acclaimed,want,shellacked,punting,mortify,segregate,coining,relax,toppling,froze,create,citing,collapsing,peeping,dapple,enticed,crinkle,describe,ferried,purifying,consolidating,clap,dirtied,occurred,climb,deplete,resulted,concocting,blabbed,reproached,ironing,blame,interweave,percolate,outweighed,encompassing,nestling,fossilized,trawl,tethering,disarm,connect,pup,swelter,contain,copying,rooting,mangled,named,mining,spill,doubted,sniffle,extraditing,double,interlocked,plunked,comforting,looped,disparaged,advancing,sprinkled,occurring,traced,glisten,bamboozled,perforated,proliferated,pupped,sobbing,looked,controlling,crackling,select,corroded,atomizing,evade,upset,certify,piloted,pain,disassociating,equivocate,overcoming,tiring,sucked,bronze,post,attenuated,feeling,adoring,engendering,slept,crisped,monitored,rafted,dumping,fleeing,gesture,swelling,bunt,wore,crating,troubled,perk,slaughter,dismissing,waxed,terminating,injected,reckoned,perform,bridle,placate,listening,ringing,belched,meditated,soiling,captivating,carbonize,auditing,bamboozle,showed,fencing,combine,hinting,plank,gill,sauntering,lubricating,molting,dressed,moaned,necking,sheathing,creaked,pressuring,tripling,funnelling,nationalized,reclaimed,raided,unscrewed,enquired,disrobing,irritate,befuddle,beckoning,conking,assailing,slashing,cub,take,hushed,rebuke,generate,screeching,fighting,jumped,deflowered,shipping,focus,awed,bet,thud,disputing,plunge,repulsed,inflated,gambled,patted,stamping,delved,detach,evolving,stacked,leaping,misappropriate,flocked,criticizing,scalded,puttied,manipulated,burdening,fawn,appropriated,lumped,procrastinating,tore,objectify,blurting,fault,regaining,taxying,vomiting,assaulting,rend,nested,forked,chlorinate,suggested,understudying,consecrated,immobilize,purchase,pronounced,hung,sniped,tilt,stopper,expected,empty,rap,hummed,typed,depopulated,approached,halted,wriggled,alluring,strode,packing,convince,wait,hybridized,prattled,retort,reverse,linking,log,slitting,dissociate,survived,billing,distinguishing,awakening,impregnating,astonish,suckling,confess,pilfer,esteeming,nesting,burgled,ask,metamorphosing,tickling,land,winning,enrich,urbanized,bleaching,zigzagging,doffed,sopped,recounting,whish,defined,metamorphose,crowning,gleaning,notched,rouge,spy,scuttling,slowing,wearying,tree,gull,dismiss,clunk,deported,thrust,outmatch,castrated,gush,toted,skulk,enacting,polling,editing,reflecting,mashed,dozing,redden,decried,drenched,participate,pierce,clawed,build,sleet,plaiting,remunerating,pinging,peeved,excavating,rekindling,described,frown,splitting,sequestered,slacked,chortled,discern,appraising,excusing,lettering,demonstrate,mangling,niggle,unionizing,tarmacked,cull,scrawling,damage,tarnishing,pumped,nestled,quoth,cabbage,rattle,dozed,honored,demolished,disappointing,rearranged,resonated,hunt,patenting,maimed,popularized,cowed,lauded,snowing,place,coddling,race,levitating,constraining,tram,risked,think,sitting,blow,roughen,unscrew,wish,distend,infected,demolishing,reply,slay,unpegged,hug,deteriorating,pause,capsize,packaging,pedaled,overcooked,answer,pitch,flitted,churning,coaxing,declining,lick,wire,abounding,lapsed,catapult,stir,shape,nose,initialled,pinking,powder,engaging,impelling,lashing,eschewing,unnerve,crooked,curdling,defecate,molesting,blurring,abbreviating,lopping,provoking,reaping,heartened,swindled,outwitted,thundered,chiselling,intimidated,faze,wheedling,confederate,mutilating,preventing,realizing,salvaging,depart,padding,porting,cemented,twittered,attaining,reiterated,whip,entertain,iodize,pinning,scaled,rumbled,clung,languished,aged,repulse,bumping,regaling,bossed,shuttle,overlapped,venturing,enjoyed,amplify,resonating,soaped,buttered,envision,trod,gyrating,smirking,dreading,iron,directing,issue,blooming,tinted,decided,haunt,charcoal,jerking,ridding,respire,untie,vacationing,perceived,fit,modernizing,perforating,surprise,acknowledged,grant,whelping,imbued,distribute,reaffirming,venture,perch,punish,firming,marking,shuddering,abuse,admired,resign,fitted,navigating,cleave,gnash,irked,worsen,forwarding,sculpted,blest,fled,debilitate,working,recall,creeping,milling,moping,detesting,rot,mystified,berthed,recited,asphyxiate,snow,poisoning,paper,seasoning,preferring,skulked,squirm,voiced,charmed,investing,constricting,say,ducked,retire,cheer,massed,haggled,bundling,rendezvousing,banish,swaddle,strengthening,slacken,inveigled,itched,invest,loaned,drowsed,sculpt,replacing,wade,jabbing,hurried,garaging,swaddled,teach,counselled,thumbed,narrating,rated,rain,channelling,distributed,mutilate,shrivel,interrelate,knit,munching,boned,croaked,exhibited,pursuing,unwind,thickening,marvelling,tracing,border,heralded,beckoned,inhaling,revelled,rotted,dusting,stultified,foresting,coarsen,confiscate,observing,exuded,blotted,has,burgeoned,lamb,drowse,slumber,surged,flit,improve,contorting,microfilm,reckon,pip,remonstrating,bloodying,tweet,debugged,slacking,swill,escape,stress,civilize,flicking,oblige,judging,rendering,pair,prostituting,curling,initiating,clambered,destroying,feud,search,lampoon,singing,unhinging,resting,march,yammer,drench,drink,gorged,arch,confabulate,flap,rank,detect,ambled,jabbering,bellow,reckoning,shriek,slap,reimbursed,grooving,ousted,ventured,comb,sour,blossom,frolicking,ossifying,guffawing,acidifying,pitying,soften,gloating,festered,frustrate,conflicting,talked,commercializing,channelled,fretting,beamed,trumpeted,binning,grasp,skipping,knighting,destabilize,canonized,sight,crave,packaged,seeded,nationalize,cocked,repaid,bowl,anguishing,seduced,gild,wavering,bantered,nationalizing,moralize,mothered,snore,valued,differing,pile,blushed,motor,imbue,shelve,melted,piling,petrifying,crew,straighten,hopping,incising,pilot,snuffling,understand,cuddled,butt,condensed,grudged,murdering,hiked,tinging,tutor,sense,lurching,allow,stemming,revered,instructing,pirate,scatter,intersperse,thaw,deformed,cowering,style,shingled,furrowing,find,refer,slump,hauled,lurched,treed,intersect,clinking,taking,explicating,frowning,gabble,wormed,steaming,cuddle,levelled,ostracize,trudge,fizzled,yoke,labelled,reheating,sink,dashing,reviving,abase,mutated,swayed,hewing,plunging,cured,sorrowed,induce,seed,lurked,misappropriated,hiccup,nodding,reddened,wet,contributed,primped,acquire,biked,boasting,coincide,owned,heighten,wrecking,willing,evicted,tussled,tingling,barged,disconnect,poke,appropriate,interlink,condone,reach,proving,quitting,straightening,nosed,tramping,deeming,emancipate,lurking,picnicking,evolved,departing,congratulate,robbing,indicate,flirted,horrify,retaliating,toting,naturalizing,sling,flowered,calk,stimulating,string,drowned,constrict,marvelled,sinned,carpeting,relish,chuckle,flaunt,veneering,tousling,atomize,grill,drown,intensify,feminizing,roll,quicken,bursting,ameliorate,inveigle,detaching,quarry,reclining,encompassed,ensued,performing,revealing,situated,smack,trudging,sojourned,distrusting,wipe,gall,dubbed,shamed,tramped,speak,slunk,squirted,proclaiming,nominate,retracting,grated,loitered,pinged,nick,scuttle,landed,drugging,seize,wheeze,modernize,skirmish,titillated,kneading,bicker,correct,perfumed,rim,react,yelping,hitch,smearing,surmounting,thrashing,desisting,despoil,slouching,asphyxiated,prohibited,recollected,prompting,dreamt,flew,mature,retired,waste,test,quaver,scurrying,sizzling,ignited,vilifying,slip,tied,purred,snatching,rallied,group,milled,affiliated,computing,reimburse,clamping,unbolting,sniffing,stagnated,cut,dot,despoiling,incorporate,withered,revitalizing,panicked,resuming,filched,consult,evolve,shading,discharging,evaporating,glimpsing,hire,blistering,propping,rhapsodizing,ripple,mute,winding,figured,denounce,journeyed,defrauded,revile,recording,differentiating,trouble,wonder,redeemed,swarming,displeasing,halter,taunt,commanding,overhung,shadowed,forcing,burp,litter,residing,enhanced,shopped,sparred,jump,paralyzed,secluded,insulting,breakfasting,outnumbered,snipping,solidifying,bayed,circumcising,hurtle,croon,turn,whacking,tallied,quibble,winter,crouched,damned,pot,hike,showering,mechanize,couple,swabbed,standardize,collate,vibrated,sending,leaved,ache,assimilated,collapsed,precipitate,tricked,imitating,intersecting,readied,dawdled,intruding,cloister,chatting,guiding,abet,gawking,overrunning,host,snicker,dread,nestle,approved,installing,coexisted,antiqued,participating,mislead,freshening,raid,ticketing,advising,beard,caning,flexing,whitewash,drooling,disrobed,slumping,devouring,fooled,reorganized,marching,scaling,screamed,administer,commingling,biking,sponge,entangled,confederating,disdaining,classified,headed,die,brush,clumping,incubate,breakfasted,ensue,spearing,curdled,charter,persevere,reaching,gulp,skewing,helping,preceded,straddling,standardizing,pack,chilled,recompense,contented,trotted,trammelled,calving,assaulted,deporting,slam,tracking,abandon,negotiate,displease,scorched,leer,lash,laminated,festooned,bond,saving,restrained,captain,relieving,witness,impeding,skedaddle,overwhelmed,filtering,mother,staring,allowed,resenting,trace,chirped,nap,rock,dredge,bargain,whipped,insulted,healing,shrinking,moistened,molted,shaping,bonded,bread,paused,backbite,pressure,dismantle,whiten,flitting,gut,portray,riddle,stigmatized,dissatisfy,nibbled,wafted,embossing,recur,saddling,look,flickered,exhale,loving,spinning,probe,rippling,woofing,exchanging,vulcanize,diffract,groaned,bang,stewed,tassel,blaspheming,sambaed,wane,feasted,desired,throbbed,plunder,knifing,dribble,redressed,weaken,apportioned,obliterating,scanning,clear,spooking,cherish,limped,resorting,proclaim,panel,sponsoring,humiliating,last,disgorged,aimed,popularizing,rinse,greasing,concerning,poisoned,drain,justifying,alarmed,capped,struck,donated,substituting,grimacing,dissipated,liquidated,animated,stun,bestride,tolerating,quavering,whooped,clucking,streaked,babble,sanction,pent,pour,survey,schmoozed,advised,steeped,quaff,watching,turning,daunt,eschewed,labelling,soldered,embroidering,smacking,lost,viewed,bartering,yapped,conditioned,equipping,wailing,recline,change,embrace,snuffled,bumped,captivate,scrimp,clashing,wondering,gouged,cox,clicking,equalizing,dwelling,materialized,mix,pat,illustrated,pawn,improvising,guaranteeing,thwacking,whisking,command,gibed,export,retch,glimmered,receiving,teaching,gilt,growled,deducting,victimize,swallowed,subside,admire,reproduce,interrelated,vibrating,flocking,descend,quarantining,terrorize,allocated,commissioned,selling,constricted,pilloried,inheriting,dug,sniggering,antagonized,shocked,riffled,housing,pirating,remove,gambolling,socking,skimming,sober,disassociate,constrain,cheeping,lancing,decelerate,differ,trawled,blocking,lock,spoon,laminating,perceive,slithered,patented,enslaved,pulp,check,moderating,faded,spruced,portion,dickering,approximate,affirmed,wavered,integrated,grease,boycotting,luring,gratified,banding,cuckoo,jug,bequeathed,lassoed,waddling,bailing,toddled,fear,step,swarmed,spied,ruminate,bustling,teaming,evened,enervating,wrote,bringing,cracking,perspiring,peeled,trusting,twanging,destabilizing,foamed,realized,recruited,incensed,imitate,regularize,arrested,twiddling,abstaining,coagulate,wrung,sapped,relinquish,absolved,shroud,nauseate,winnowing,dive,reorganize,shouted,moo,splice,splattered,lower,disseminated,join,upsetting,regaled,forage,chronicle,begetting,sensed,shepherd,darkening,bedded,register,believing,framed,polarized,woke,drizzle,feel,pluck,sponging,drill,adhered,acclaim,split,parted,pinch,degenerate,compressed,dehydrated,journey,guzzled,smooth,insured,vacated,lined,celebrated,mow,scoured,tunnelling,barking,disarmed,patrolling,paralyzing,bottling,spin,imbibe,photocopy,sneak,felicitate,translating,pearled,stay,exact,harry,stroll,zoomed,whelped,unhooked,top,boasted,fuel,sanctioned,bonding,balance,wagged,anointed,mellowing,ascertaining,cast,poll,preaching,entrap,capture,crimp,pawning,encouraging,dissuaded,invert,disgraced,inscribing,smarten,soothe,twitching,luxuriate,engendered,leafing,illustrating,squashed,surpass,snarling,transfer,stalled,justified,force,hoot,spun,eliminate,consulting,nutting,completing,pith,stencilled,distinguish,begot,arouse,primping,salivate,coped,sacrifice,affirming,rating,grace,sprayed,crooking,baste,hoodwink,meander,trill,blunted,snored,repel,hurling,clerking,gasified,toast,burst,bending,gobble,resound,flabbergasting,wishing,veining,doctored,loading,surrendering,scalding,filing,yammering,intimated,bundled,smoke,surrender,losing,diluting,proceeding,disheartened,shellac,sneer,scudding,inventing,camping,outmatched,relieve,chaperoning,conjecture,rekindle,reviewing,grabbed,seated,stow,lusted,skinning,jolt,echo,caramelized,converging,dismember,romping,eliminating,yield,gracing,imply,plagued,flare,damaging,exist,scoot,belittled,purified,paraded,collude,bestrode,transpiring,drugged,chip,energize,swirl,bowled,tinning,mutter,dicker,praise,commented,manufactured,dabbing,recorded,clawing,redressing,oscillated,burden,stared,stringing,squeak,spoilt,pinched,pipped,suckled,emulsifying,ridicule,roasting,squeezed,bewail,gibing,rinsed,poison,mew,point,shelled,snarled,attacked,swigging,disturbing,smiling,eliminated,enthused,bequeathing,come,speeding,defrosting,fleece,annexed,victimized,jammed,intoned,shrivelling,ceased,lobbed,negotiated,steep,deport,butchered,acquit,intertwine,shrouding,stripping,thundering,cling,weighed,trading,coveted,screen,deposing,genuflected,canvassed,panting,owe,disgracing,neutralized,expect,shooting,notified,reacting,panted,pelting,assured,bedding,feminize,kidnapping,slaughtering,squirting,functioning,shrieking,replace,dismembered,repress,splotched,scandalizing,dismissed,plummet,prefer,characterized,grumbling,refereed,bus,poached,pegging,scavenging,plunged,deride,whimpering,worsened,pity,wave,pitied,dice,uncoiled,stumbling,grunting,magnify,rejuvenating,reverting,drydocked,follow,sanded,present,annoyed,beautified,dawned,raved,submitting,fluctuating,sanding,traipsing,mechanized,affiliate,receded,dirtying,measured,shambled,dispersing,promote,warbled,prospected,wheedle,berry,masticated,yelled,sin,dyeing,rupture,treasure,inundated,flock,befuddled,crisscrossed,forfeited,officiate,doctor,tortured,tailoring,snuck,trailed,jailing,orphan,lambing,ascertained,flattening,defecating,eulogizing,knocking,head,regularizing,fret,riled,bruise,transported,filleting,ruffled,stultify,revealed,whittled,tolling,portrayed,proved,integrate,haunting,touch,painted,go,nibble,shackling,toddling,tried,bash,help,delivered,frolic,sizzle,counsel,ransack,enforcing,nudging,tilting,selected,beached,repeat,decreased,prickled,swished,leaning,scooting,salted,overawe,clacking,compensate,caged,switching,spurt,fattening,moistening,heat,antiquing,straddle,drizzled,federate,fancied,ripened,spooked,transmuted,hoing,contributing,contorted,cripple,deprived,heralding,extracting,racing,despise,cork,perching,concurred,extirpate,folded,groan,reopened,dope,widen,stammer,fowl,assessing,clubbing,burr,spool,incriminated,debated,compelling,slackening,amalgamated,bagged,stuttering,immerse,forking,slipping,muttered,dining,duping,eddy,griping,overcome,delve,planted,devastated,emanated,baptized,conducting,tattoo,rivet,mellow,camouflaged,weep,agglomerate,fazing,nest,purse,burdened,teethed,kindled,compromised,grew,cleaning,slogging,daunted,depressurized,malign,scald,suspect,squat,depopulating,grafted,concentrating,disbelieved,hurtled,trebling,threatened,tussling,greet,chortling,deluding,purr,degenerating,swiped,calibrate,knife,injure,pressed,mail,fleck,fumbled,shred,moralizing,debarked,disquieted,backbiting,blindfolding,mesmerize,contrasting,visualized,pulled,issued,venerating,impressed,splattering,reviewed,wailed,disembarking,obtaining,hammering,dressing,fillet,overreaching,pitching,disconcerted,unsettled,portraying,betted,wash,distracted,triumphing,broken,fall,brain,phone,joked,accused,trickled,plaster,basting,subtract,seclude,recognized,pulverize,crewing,accompanied,gripe,bulged,excavated,lecturing,dispossess,snatch,retrieving,drifted,daub,autographing,dissipate,sniping,brand,awaken,antagonizing,outbid,club,clustering,revive,wrinkled,endowing,admit,dry,curdle,pelt,lacquer,piercing,mince,merged,deafen,apportion,steadying,picture,wake,bother,gashed,testing,kneeling,calumniate,cringe,peeling,humbled,falling,housed,paddled,root,stoned,peer,flour,projecting,droop,dislike,eulogize,overstepped,desiccate,binned,sound,barge,gloried,eventuated,occur,outstripped,recoiled,stretching,rake,charming,dust,scratched,thin,belittling,enthusing,smoothing,moor,abandoned,lugging,yammered,scandalize,chauffeured,distended,scout,extruding,gained,stand,standing,opposing,doodled,defraud,yoking,forbade,overheated,graduate,adorned,quarrelled,soared,struggled,perambulating,confiding,stab,mooned,nailed,dallying,scented,vitrify,cuckolded,haunted,drenching,hefting,gloving,mooring,secured,wed,irritating,surpassing,worshipped,presenting,ventilate,played,caressing,pouring,hoarding,reinstate,smudge,fogged,ruined,prospect,prized,profess,blocked,range,ingest,conjure,discombobulating,pinioned,pattered,expounded,gleamed,wobbling,shorted,ruin,airmailed,lunched,hearing,lump,survive,tiling,unifying,narrowed,rocketed,glistening,blackmailing,elope,unnerved,prowl,sniggered,gladden,wearied,conjoining,wink,grouching,class,conflict,filter,advance,carol,rode,compensated,slimed,unbuttoning,smarted,cadge,augmented,languishing,ferrying,gulling,pictured,moan,dented,bubble,consecrate,shell,abetted,phoned,crisscrossing,pacify,voted,digging,stall,beware,extradite,plow,deprecated,regretting,squelching,lurch,misleading,chlorinating,lamented,loosening,strangle,parboil,gulping,modulating,hayed,contaminated,skyrocketed,rescuing,hazarding,vaporize,cropped,polished,propped,commence,vote,interlinked,elate,hobnob,denounced,had,abstracted,jam,burnt,striding,shun,simpering,overhear,elaborate,interpret,perish,goggling,desalting,portioned,flapping,fizzing,tussle,diverge,forest,ceded,condemned,glove,simper,plodding,knowing,fixing,lathered,spending,kept,shod,succeeded,supposing,circumcise,invaded,proliferating,thronging,undoing,armed,bled,infect,shoving,bow,declared,gobbling,balancing,dangle,scanned,begin,analyzing,coast,waved,sprawl,bustle,own,cowered,toboggan,upbraid,weary,scribbled,lift,bleated,signal,lengthening,bribe,capping,chalk,died,hearten,accept,gestured,spiced,attempt,tug,shackled,castigate,trudged,assessed,underlay,logging,needing,twirled,detonating,pepper,naturalized,discard,embalmed,snacked,failing,outlawing,kick,sifted,stored,ploughing,disengaging,dampened,stayed,appalled,absolve,loom,cultivated,rationed,fainting,drydock,inflame,handcuffing,clogging,crow,halting,precipitating,cremated,clattering,clothe,piqued,loping,hitching,trail,dinging,end,reading,expired,conserved,ferment,vanished,marinating,bicycle,set,freshen,superimposing,preached,recount,mop,primp,tire,sparing,recreating,stigmatizing,riling,unbuckling,lop,sagging,broaden,engaged,cluck,sewed,bleat,wringing,astonished,procuring,taste,haul,chagrined,jar,fracture,elevate,overburden,spoil,attired,identifying,mosey,unchain,deflate,fling,stomping,chide,flush,concurring,ping,yap,squashing,swindling,motorize,wasting,deluged,wagered,misspell,wakened,filch,cladding,thicken,roamed,excoriating,zip,pried,interlace,flutter,annex,cultivate,omitted,deriding,floating,peddling,worship,rupturing,disentangle,buttoned,canoeing,shoved,coupling,trooped,continued,cooling,exited,communicating,apprenticing,crabbing,explore,sup,dumped,tinged,dazing,cloistering,decrying,hesitating,tar,pollute,terrify,elude,immunized,discontinuing,nurse,believe,scintillate,created,seeped,edging,depended,ruptured,straightened,wasted,smoothed,minding,baked,laughing,delude,radioing,squatting,disconnecting,prepare,depleted,fowling,wagging,scallop,lug,urging,breathing,snap,conjecturing,slung,bronzing,surprising,whitewashed,bloomed,salting,stalk,dissatisfying,fascinating,redrawing,pirouette,recast,embezzled,brightened,contracting,sprouted,cram,refreshing,wedding,giggled,purposed,distilled,enlarged,clinging,voyaged,coagulating,solidified,cashing,jest,scrimped,suppress,insisting,soar,butchering,radiate,taxed,maiming,warned,cellar,thriving,billeted,balloon,widowed,rendezvous,learning,building,whinnying,moseyed,squishing,taming,pasteurized,jet,pry,heel,squabbled,amalgamating,prickling,complete,rely,dripped,navigated,kindling,gladdened,vie,puked,gushed,existing,eloped,slid,apprenticed,found,normalizing,upbraided,dirty,horrifying,furrow,deny,shatter,spotting,overhang,regained,fetch,cluttered,topped,hovered,dedicate,dislodge,hungering,lusting,censure,feathered,undress,distilling,swim,gibe,mint,screaming,letter,bridge,lobbing,styled,repaying,wince,wrested,dodging,descended,waited,shivered,initial,dangling,hoping,chromed,shot,murmured,tarrying,tighten,garlanding,samba,plated,recoil,cramming,illustrate,harvest,streaming,unlatched,correspond,served,telephone,bopped,confused,prattle,blaming,lingering,damn,spewing,yellowing,thrive,covered,reheated,tag,crying,placing,satiate,riddled,suffocated,unpeg,flex,continuing,crouch,crane,decreed,jerked,quacked,snacking,melting,forfeiting,obsessing,stupefying,soaking,noted,broadening,towing,swarm,registered,thank,furnishing,disappear,screwing,inherited,reconstituted,fired,drift,stressing,blurted,browning,ruling,opening,redraw,undercharging,chime,leaven,wrenching,saturating,imprison,silence,instructed,spelling,drinking,carve,gesturing,bite,praised,coined,inlay,fined,wielding,hunting,responded,discontinued,scorn,thawed,wreathed,bluff,worry,chiming,snorting,fasten,quarrelling,repay,adopt,scrambling,bathing,tittering,reproduced,devastate,cuckolding,disgruntle,disbursed,depreciate,weaving,transferring,erect,extract,squawking,pirouetting,weld,withering,worshipping,dickered,cudgel,granting,behaving,manufacturing,incriminate,painting,relayed,inebriating,deposited,wallowed,nominated,blind,cow,deteriorate,hove,culling,seasoned,soaping,courted,greeting,cream,dim,converting,suffer,mopping,befalling,rendered,inquired,rimmed,reiterating,scaping,sag,cluster,cock,eradicate,blistered,slouched,divested,herald,deforming,accruing,pummelling,tense,tapped,skittering,sparring,pained,coax,mating,rumpled,hastened,immobilized,hew,matched,pined,braising,decreasing,calve,cackle,mount,neaten,misappropriating,schlepped,depreciating,hardened,smartening,laminate,estimate,clipped,sowed,sanctioning,despaired,drooled,stepping,smudged,lanced,switch,christened,disclosed,announcing,spooling,quit,steering,televised,paid,coinciding,condoning,plagiarized,interlock,mobilizing,granulate,denuded,faltered,slumbered,riffle,decimated,assimilate,griped,mist,lasting,believed,knocked,rendezvoused,ogle,contained,compressing,handing,ushering,sprinkle,transferred,examining,redress,silencing,destabilized,federating,pronouncing,slow,rimming,flog,transcribed,toiling,interlocking,assuage,scrounged,underlying,prohibit,clashed,considered,diffuse,slobbered,chinking,shrugging,twirl,categorize,allured,imagined,plowed,glance,swerving,utter,filled,sheltered,mushroomed,slice,arising,throwing,regretted,agonize,deal,dating,please,herded,notch,isolating,tainting,commending,consort,hasten,fondled,proposing,chat,visualizing,sniffed,waggle,bob,bring,ranging,thinning,silver,churned,acclaiming,applauding,bowing,intimidating,stippled,designate,brushed,embellishing,embellish,debating,cooperating,muddled,posted,creaming,spit,pedal,corralled,clam,laughed,disembark,inhale,chomp,stripped,solace,sterilizing,avoiding,scampered,deriving,grinding,circled,infuriate,gleaned,distorted,decentralized,lean,pump,ended,speeded,brewing,weigh,extend,flickering,intermixed,bathe,boating,raise,mind,mooed,kidnapped,classifying,inlaying,clacked,replenishing,said,drove,disintegrated,astonishing,cheep,skip,twined,monitor,perplex,autograph,quarrying,berried,nut,jibe,erupting,audited,crucify,appoint,suffuse,relate,enamel,ached,smacked,explain,liberating,gallop,relinquishing,docked,pairing,doping,frustrating,rousing,lap,despairing,boarding,sundered,slicing,navigate,vulcanizing,loan,degrading,frolicked,towed,stumped,teem,propose,chanted,doddering,floor,built,pitted,stretched,stray,gyrate,shredded,obliterate,gaping,sleeting,sniff,boat,sunbathed,hobnobbing,frosted,modulate,polled,grip,weave,refrained,urge,annihilate,shoo,ground,twanged,seat,loitering,stung,transcend,clambering,calcifying,galloping,glancing,unpegging,atomized,crinkling,compel,unroll,affect,glint,lighting,vex,sadden,rasping,broadcast,demoralize,pushed,elevating,pant,electrocute,clatter,reflected,expelling,embittered,smartened,enact,skyrocketing,galloped,load,stratify,dribbled,prophesied,provided,convulse,frightening,inscribed,terrified,booked,delighting,tarring,deified,celebrate,ordaining,envy,perceiving,motored,comfort,read,pasted,gripping,represent,stop,bone,scarring,risking,planned,afflicted,chant,bat,abridged,cooing,dinged,consume,policing,magnetize,garaged,orphaning,grouch,yawn,disappointed,lying,ferreting,piloting,calling,decayed,made,light,reducing,skipped,discerning,clamming,manicure,slash,glistened,doze,faint,sneeze,perched,examine,cavorted,stumping,pawing,keep,demobilize,touring,musing,delivering,attached,inquire,creased,burning,petted,spouted,disbursing,stock,choose,dialled,rocketing,demonstrating,smelling,persisted,exhibit,fabricated,purged,engage,holidaying,yelped,forging,ordain,whined,coveting,grouse,show,depending,bisecting,labour,effaced,knitted,prompted,bifurcate,excelling,ladling,glimpse,listen,sweltering,rescued,trailing,smirk,vitrified,benching,stable,cause,fertilizing,cooled,dashed,disgrace,tensing,translated,presented,frost,exhaled,tricking,bore,protest,ostracizing,lengthen,apprentice,captaining,extirpating,mulch,unfolding,copy,oyster,put,plumb,assisting,disarming,undo,gasify,curtain,quizzed,gawked,clattered,imbuing,squeeze,regularized,glazing,salvaged,capturing,limiting,shampooed,warmed,carved,hocked,shed,mooning,convened,draping,wriggling,flared,depicting,striking,cocking,circumvent,harassing,predicting,ordered,inspired,broil,schmoozing,jiggling,reversed,germinating,inlaid,crook,engender,proliferate,bragged,loosed,nuzzle,agreed,enchanted,abstained,distressed,necked,crinkled,relied,submitted,deliver,regale,groove,pinching,cluttering,cooperate,caulked,steal,concealing,admitted,related,unhinged,anguish,devour,bequeath,drawled,short,spend,resigned,yielding,cane,meditate,bombard,rustling,commiserate,notify,conserve,dictate,combed,brag,soothing,gnawing,molested,flabbergasted,wag,bulging,deemed,putty,condensing,condition,scurry,suspending,televising,puzzled,guessing,appeal,whimpered,scan,consenting,plagiarize,brick,abstract,muddying,pummelled,wedded,free,cycling,recreate,hated,beaching,suckle,faced,reporting,leaking,frighten,appending,pasturing,jog,chug,reveal,provide,branching,cored,toppled,outnumbering,rent,loomed,ornament,aggregating,metamorphosed,interwove,sterilized,mumble,flowing,dissenting,lodge,glared,scud,disconnected,retreating,roar,bussed,shift,interchanged,toughened,unlaced,smirked,echoed,acted,compensating,nicked,tobogganing,hoard,enlarge,looping,took,unlocked,suppressing,coated,bridling,tiptoe,slugged,ventilated,smuggled,decelerated,sailing,ejecting,bristled,bathed,etched,received,approximating,swing,promoting,pitched,immolated,prickle,chuckled,exuding,veiling,braining,ceasing,courting,installed,consolidate,retrieve,flaming,rummaging,prevailing,mingling,discovering,irritated,truck,stubbed,pet,shredding,plummeted,canoed,grind,coalesce,conk,finish,upbraiding,blindfolded,dallied,grimaced,polluted,desiccated,chalked,establish,perfume,swiping,guided,valuing,bopping,latching,judge,assigning,cracked,drone,mopped,justify,overheard,unsettling,traversed,jeering,squelch,marked,imbibing,snowed,press,composed,tiled,invigorating,chanting,resorted,arched,enforced,excelled,ostracized,whale,handcuffed,quadruple,bridging,dampening,percolated,streak,bordering,confirm,scandalized,devalue,logged,blaze,lambasted,longed,lining,streaking,boom,splintering,doubled,shoe,deteriorated,crystallized,plummeting,munch,disdain,readying,baffled,urinated,serving,age,pealed,holler,diaper,abased,tarmacking,strung,flourished,rust,skedaddling,pattering,cuckold,secularized,totalling,trying,receive,flooring,picnic,edge,struggle,commission,disentangled,conjured,pocketing,grappled,scooped,troubling,crawled,discuss,stuck,expressing,sealed,flop,water,fleeced,overburdened,graze,peddled,scorned,commercialize,pushing,spluttering,spellbind,interview,dismantling,gambling,paired,approximated,sleepwalked,drive,razing,snubbing,stocking,graduating,allotted,humbling,snuffle,appreciated,voyaging,know,trotting,undulate,wintered,widened,gag,dupe,crystallizing,converted,emblazoned,stabbed,finishing,shaved,hid,cavort,trapping,tint,hesitate,lulled,telling,overcharge,curtained,leafed,sugared,appreciate,chomped,caddy,screening,thrumming,level,freshened,fire,recurred,disorganized,jousted,mobilize,objected,jabber,earn,spraining,tightening,culled,interested,clicked,scrutinize,rained,abduct,advanced,revert,upgrade,exposed,tacking,wield,knot,guarding,dissented,assigned,fertilize,peek,rounded,defame,scurried,specify,exercising,satiated,shone,subtracting,glittered,mystify,cheapening,neatened,quake,operated,parboiled,institutionalized,picnicked,differentiate,joking,rumple,gum,learn,starve,tottering,blasting,grafting,view,discarded,wintering,swimming,lacing,remark,scuffle,cascade,tipping,divorce,grieved,festering,slopped,quote,fooling,blanched,polymerized,straggle,discouraging,ensuing,hint,suffusing,bill,tolerated,vacillated,duelling,rumbling,wept,edited,reunite,manicured,admitting,daubing,dull,magnetizing,hoarded,trebled,plopping,wrestled,simmer,wring,exhilarated,regarded,skitter,rustle,scrutinized,prove,rut,bawling,aggravated,drilling,tousled,called,japanning,looming,foreseeing,gurgle,arranging,alternated,assign,scalloping,nobble,cuffed,telegraphing,relaxing,diffracted,batting,killing,paling,toll,minced,genuflect,seeing,vomited,bargained,belching,loathing,fester,enlightening,jumbling,hitting,reiterate,cheering,shackle,ingested,crackle,gleam,define,disparaging,scuttled,glued,milked,tidy,crucifying,jab,rippled,thatched,hiking,wow,punched,whooping,decaying,tittered,welded,gloat,snubbed,steepening,disapproving,liking,fluoresce,disgruntled,stammering,toasting,feed,mounding,quavered,commencing,choreographing,cloud,lugged,parachuting,penalizing,channel,shuffling,interesting,dismount,menace,blat,surpassed,vaporized,diagnosed,narrow,seeping,lofted,creamed,cavorting,undulating,depose,disillusioning,book,moulting,plonked,blunt,cackling,trawling,glimmer,toughen,maligning,manage,compiled,saunter,deepen,shelving,diluted,shunting,suspected,scrawled,pelted,mated,deflated,threatening,alter,denying,trolling,neatening,broiling,counting,audit,snigger,stifling,semaphore,scintillating,imbibed,promulgating,shepherded,magnetized,subtracted,staged,shrink,dominated,boiling,graft,glowered,teemed,married,gather,incriminating,eroded,compute,uprooting,quibbling,bag,plunk,shark,chinked,astounded,shimmer,scrabbled,chucking,lurk,unzipping,scuffled,prophesy,answered,fuming,calm,soldering,quietening,steer,inebriate,fought,trained,puffed,penalize,terminated,theorizing,dreaded,pausing,give,bifurcated,understudy,unleashed,flustered,disassembled,gliding,hypnotized,fight,resonate,greased,reinstated,briefing,enslaving,loiter,entice,iodizing,toot,flecked,clench,coexisting,cope,disgust,oozing,burrowed,gnashing,oar,plastered,engrossing,desecrating,envisage,surveying,convey,detonated,correlating,accepted,riddling,study,caulk,crouching,publicizing,skewer,imprisoning,shepherding,leaving,sweep,banquet,exult,drool,bruised,abandoning,modelling,slander,rooted,jeered,transpire,revel,positioned,nominating,clad,tasselled,perspired,bounced,reappearing,quickening,airing,cross,pecking,collaborate,whispering,organize,voting,ducking,hang,wangling,starving,impose,flatten,frightened,bludgeon,screw,policed,need,coincided,snoozing,disgusting,sneering,recommended,turfed,waddled,repressed,bilk,excoriate,lumping,rasped,budded,hurrying,flogged,nettled,lapse,erecting,decentralizing,exclaim,slurped,grouped,jut,repudiating,debarking,muted,hunching,scrounge,warehouse,boarded,encircling,rearranging,held,pooling,eye,hatching,thirsted,season,doubt,aid,quaffed,weeded,hurt,restrict,grappling,eradicated,heaping,troll,hoisted,obliging,weakening,latched,tracked,bray,drew,vacuumed,retorted,charcoaled,purloin,wean,shuffle,lull,heeled,advise,noticing,concocted,interlacing,multiplied,fry,undertaking,evaded,preferred,frayed,commiserated,locked,trumpet,collaborating,detached,concentrated,squabbling,raked,obliged,muddle,drank,grounding,discover,waxing,indicted,exalt,sap,noshing,flexed,bouncing,galvanize,baa,hazarded,energizing,clog,polka,vied,exciting,wheezed,twirling,debugging,plowing,televise,blazing,manufacture,popped,incubating,excommunicate,neglecting,embezzle,tipped,bawled,reaped,fluctuate,lauding,cohere,curing,increased,signed,swelled,correcting,intimating,dodged,shocking,flick,spent,pink,band,microfilmed,prospered,rummaged,coxing,murmuring,confining,restored,predominating,evacuated,sleep,distract,plumbed,abbreviated,flash,stipple,sheared,lassoing,detest,amplifying,expanding,conciliate,heartening,sliding,chalking,gain,foraging,tooting,thirst,hatched,fancying,groused,fly,appear,kennel,augmenting,seeking,quizzing,dropped,baying,fraying,boast,gilled,swore,plonk,stalked,styling,roosting,struggling,quaked,caned,thanked,chisel,agitate,climbing,sketching,stream,laboured,shunning,wolfed,shouldering,switched,cleaned,spooning,surmised,grudge,meant,vein,maintain,setting,acknowledge,resided,ironed,unified,chastising,tapering,frying,promise,probed,bearing,collaborated,scenting,shuttered,crowing,hybridize,enrolled,purify,thump,welcoming,lowering,ravishing,hinder,sprawling,spurting,ballooning,try,slate,cop,hauling,lessening,garlanded,will,toiled,glitter,assailed,conspiring,snuffing,warped,stone,sat,shooed,dreaming,classify,spreading,stunk,currying,groped,dappling,stewing,exacting,ferry,graced,eaten,ripen,crumpled,collected,staff,hock,exasperate,startle,coo,moon,recompensed,mushroom,yanking,stressed,billeting,ready,zoom,etching,pursued,swinging,frothing,began,tramp,detested,hooting,pilfering,lathering,stack,fidget,taint,sullying,motorized,pardoned,extrude,grieve,saluted,aggregate,appreciating,relating,pranced,crammed,placating,tutoring,bike,start,tread,seal,reaffirmed,catnap,dress,hack,torture,electrifying,sign,spilling,supply,clerk,isolated,inundate,beguile,wrest,dissatisfied,pearling,cheated,promenade,prompt,gasping,disconcert,supping,clumped,skippered,remonstrate,whirl,cement,disillusioned,jumble,soured,standardized,peeped,brown,glorified,yowling,sired,attenuating,interspersing,branched,preoccupying,chided,memorized,hate,objecting,entertaining,divided,wander,intensifying,bustled,neglected,ruining,revolting,intrigue,luxuriated,spindled,protruded,forgive,tamping,swallowing,cremate,expunging,tested,bawl,gorge,schlepping,sire,tap,bleed,altering,belittle,percolating,snare,stamped,vexed,lubricated,operating,function,embracing,sparkled,ranked,started,speckling,pottering,shifting,winkle,dislodged,journeying,ladled,make,developing,warbling,accrue,unlocking,depopulate,assuring,snoop,discharge,foraged,sinning,remembering,shoeing,crimson,lapsing,flying,rouged,splinter,scouting,amuse,segregating,lapping,glory,doffing,bewildered,fork,unfastening,veer,wear,snuffed,usher,flipped,wager,invading,boogied,wound,suffered,tickled,telegraph,bagging,scattered,expectorating,comparing,hosting,motivate,spawning,combated,voyage,fondling,demoralized,dominate,levitate,ripening,hiccuping,antique,merging,nod,pacifying,spellbound,neighing,swaddling,slop,deifying,volunteering,filtered,redeem,weight,wobbled,stepped,spurring,defecated,polarizing,depressing,glare,evaluating,abrading,stratifying,breading,price,bury,toughening,amalgamate,perplexing,drop,coaxed,splay,danced,interweaving,sticking,intriguing,canter,gossip,rejoiced,miff,distributing,constrained,continue,indict,daubed,chortle,grope,furrowed,beckon,reached,wangle,stoppered,scribble,marched,tallying,cellared,decentralize,dulling,earning,square,distrust,defrauding,button,disputed,twisted,chancing,conceded,cadging,granted,performed,yearn,confine,frisked,drowning,pioneered,edged,marrying,differed,chewed,photographed,wondered,cloistered,praying,outstrip,coming,weakened,valet,tog,carpeted,briefed,atrophy,mask,mystifying,barked,sparkling,mooing,uncoiling,chauffeur,glazed,thickened,laugh,deice,endure,providing,flourish,reverberated,converse,sketched,chipped,proffering,abate,fawned,ousting,handcuff,bewitched,rankle,commiserating,dawdling,giggling,making,impeaching,caging,recounted,iodized,sledged,shamble,smart,coiled,mumbled,masked,regarding,satisfy,restrain,skim,wallpaper,obligate,unfastened,scoop,approaching,pit,amassing,stagger,buzzed,intermixing,rove,aroused,buying,accusing,uncoil,weekend,call,leapt,excised,act,horrified,transform,hammer,dam,replied,vaulted,crooning,snub,irking,fructify,mantled,perusing,pencilled,howled,shudder,feuding,finished,abhorred,cure,crab,experiencing,leach,roiled,alienate,characterizing,diverting,ting,spellbinding,purchased,urbanize,gauging,shelter,reproducing,disappoint,whished,crawl,reward,supped,rocked,softened,throw,frisking,occupied,extending,folding,dedicating,wafting,match,observe,dent,abutted,harmed,twine,decreeing,greeted,understanding,procrastinated,polluting,filching,prizing,calmed,position,voice,lust,realize,gluing,purloining,shook,recite,dawdle,stratified,attach,smuggling,spanning,rested,blaring,equivocating,portioning,urbanizing,overcharged,tilted,conspire,whisked,depressurize,sweetening,pursue,jetted,trembled,transport,contenting,tooted,moseying,puke,vacation,rejoicing,hay,refereeing,shuffled,chaperone,expecting,slapping,crayon,challenging,operate,potted,corrected,procrastinate,rising,deplored,hushing,urinate,romp,existed,lance,perked,trammel,intoxicate,nuzzling,abolishing,soap,rapping,demonstrated,learnt,terrorized,glean,worming,reunited,understudied,acting,guzzle,revelling,upgraded,stride,silenced,stabilize,masquerade,trekking,proclaimed,impede,tantalizing,expelled,black,basted,baffle,wrench,squealing,neutralize,whaling,slaughtered,break,luxuriating,pick,arrest,blur,spliced,soaring,riffling,confiscated,crowned,award,counselling,castrate,darted,taping,deck,occupy,spooled,package,mollify,stick,romped,sneezing,putter,beguiling,raking,wearing,dappled,comforted,bombarding,dreamed,collided,cared,disapprove,arrived,mellowed,anglicizing,suspend,mulching,massacred,confined,blitzed,swabbing,converge,embalming,engross,complain,cruise,recommend,send,bossing,peep,managing,awakened,manacle,investigating,massacring,decked,veering,fascinate,secularizing,bickering,crunch,grimace,clip,scaring,converged,transcended,skippering,overawed,engrave,waft,foaming,anchored,mushrooming,compete,fail,type,stalking,derided,hooted,sovietize,dismounting,visited,slapped,treading,wolfing,arrive,heft,dodder,amble,grapple,dismantled,assassinating,assimilating,misted,review,stippling,thrash,discomfit,twinkle,oppose,lumbering,rifle,vacuuming,deflower,disillusion,cultivating,trilled,imagine,overheat,assuaged,towelling,twang,interacted,lodged,beep,clamber,grudging,deserted,cooking,paw,atrophying,tousle,bullied,chastened,idolized,hop,whoop,zing,overturned,chuck,oil,swam,excused,adopted,magnified,surmising,hatch,deceive,boogieing,hallucinate,blowing,smarting,retorting,waken,disfiguring,sleeping,sheathe,failed,cover,ram,feared,coarsened,weighing,staying,circumventing,petting,loathed,yank,tidying,vacillate,blunting,esteemed,flatted,transpired,segregated,scooted,parch,deviate,stiffening,swilling,noticed,promenading,flee,exploring,ignite,cower'.split(',')

	print('Word whitelist contains {} words!'.format(len(whitelist)))
	import math
	for log_sppmi, sppmi in zip([0], [0]):  # zip([0, math.log(5), math.log(10), math.log(20), math.log(40), math.log(100), math.log(250)], [0, 5, 10, 20, 40, 100, 250]):
		for pmi_type in ['ppmi']:
			for cds in [1.]:  # [1., 0.75]:
				for window_size in [2, 5, 10]:  # [2, 1, 5]:# [5, 2]:
					print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={}; ...'.format(
						pmi_type, window_size, cds, sppmi))
					transformed_out_path = os.path.join('/disk/data/tkober/_datasets/gigaword/', 'wort_vectors',
														'wort_model_ppmi_lemma-False_pos-False_window-{}_cds-{}-sppmi_shift-{}'.format(
															window_size, cds, sppmi
														))
					if (not os.path.exists(transformed_out_path)):
						cache_path = os.path.join('/disk/data/tkober/_datasets/gigaword/', 'wort_cache', )
						if (not os.path.exists(cache_path)):
							os.makedirs(cache_path)

						vec = VSMVectorizer(window_size=window_size, min_frequency=50, cds=cds, weighting=pmi_type,
											word_white_list=whitelist, sppmi_shift=log_sppmi, cache_path=cache_path,
											cache_intermediary_results=True)

						vec.fit(giga_reader)

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
	p = '/disk/data/tkober/_datasets/wikipedia/corpus/wikipedia_utf8_filtered_20pageviews_noid.csv'
	wiki_reader = CSVStreamReader(p, delimiter='\t')

	#out_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'wort_vectors')
	out_path = os.path.join('/disk/data/tkober/_datasets/wikipedia/', 'wort_vectors')
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
	for log_sppmi, sppmi in zip([0, math.log(5), math.log(10)], [0, 5, 10]):#zip([0, math.log(5), math.log(10), math.log(20), math.log(40), math.log(100), math.log(250)], [0, 5, 10, 20, 40, 100, 250]):
		for pmi_type in ['ppmi']:
			for cds in [1., 0.75]:#[1., 0.75]:
				for window_size in [1, 2, 5, 10]:#[2, 1, 5]:# [5, 2]:
					for dim in [25, 50, 100, 300]:
						print('CONFIG: pmi_type={}; window_size={}; cds={}; shift={} dim={}; ...'.format(pmi_type, window_size, cds, sppmi, dim))
						transformed_out_path = os.path.join('/disk/data/tkober/_datasets/wikipedia/', 'wort_vectors', 'wort_model_ppmi_lemma-False_pos-False_window-{}_cds-{}-sppmi_shift-{}_dim-{}'.format(
							window_size, cds, sppmi, dim
						))
						if (not os.path.exists(transformed_out_path)):
							cache_path = os.path.join('/disk/data/tkober/_datasets/wikipedia/', 'wort_cache', )
							if (not os.path.exists(cache_path)):
								os.makedirs(cache_path)

							vec = VSMVectorizer(window_size=window_size, min_frequency=50, cds=cds, weighting=pmi_type,
												word_white_list=whitelist, sppmi_shift=log_sppmi, cache_path=cache_path,
												cache_intermediary_results=True, dim_reduction='nmf',
												dim_reduction_kwargs={'dimensionality': dim})

							vec.fit(wiki_reader)

							if (not os.path.exists(transformed_out_path)):
								os.makedirs(transformed_out_path)

							try:
								print('Saving to file')
								vec.save_to_file(transformed_out_path, store_context_representation_matrix=True)
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


def perform_context_selection(input_path, output_path, num_contexts):
	print('Processing {}...'.format(input_path))
	wort = VSMVectorizer.load_from_file(input_path)
	print('Wort Model loaded!')

	print('Performing Context selection...')
	wort.fit_context_selection(num_contexts=num_contexts)
	print('Context selection done!')

	print('Storing...')
	wort.save_to_file(output_path, store_context_selection_matrix=True)
	print('Done!')

if (__name__ == '__main__'):
	args = parser.parse_args()

	#test_token_and_vocab_count()
	#vectorize_pizza_epic()
	test_pizza()
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

	# Load experiment id file
	if (args.experiment_file is not None):
		with open(args.experiment_file, 'r') as csv_file:
			csv_reader = csv.reader(csv_file)
			experiments = []

			for line in csv_reader:
				experiments.append(line)

		if (args.experiment_id > 0):
			logging.info('Running experiment with id={}...'.format(args.experiment_id))
			experiments = [experiments[args.experiment_id - 1]]
		else:
			logging.info('Running all {} experiments!'.format(len(experiments)))


	#'''
	#vectorize_wikipedia()
	#vectorize_gigaword()
	vectorize_1bn_word_benchmark()
	#vectorize_amazon_reviews()
	#vectorize_ukwac()
	#vectorize_wikipedia_epic()
	#vectorize_bnc()
	#print('Running BNC samples...')
	#vectorize_bnc_samples(input_file=os.path.join(args.input_path, args.input_file), output_path=args.output_path,
	#					  cache_path=args.cache_path, current_sample=args.current_sample)
	exit(0)

	#print('Running context selection...')
	#for in_sub_path, out_sub_path, num_contexts in experiments:
	#	perform_context_selection(input_path=os.path.join(args.input_path, in_sub_path),
	#							  output_path=os.path.join(args.output_path, out_sub_path),
	#							  num_contexts=int(num_contexts))
	#exit(0)

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
