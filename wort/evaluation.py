__author__ = 'thomas'
from argparse import ArgumentParser
import logging
import os
import random

from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm import *
import numpy as np

from wort.datasets import DATASET_FETCH_MAP
from wort.vsm import VSMVectorizer

# TODO: Selector for datasets
# TODO: OOV mode ('ignore', 'random', 0.5)
# TODO: replace NaNs
# TODO: Some kind of cross-validation
parser = ArgumentParser()
parser.add_argument('-i', '--input-file', type=str, required=True, help='input file')
parser.add_argument('-ip', '--input-path', type=str, required=True, help='path to input file')
parser.add_argument('-e', '--evaluation', type=str, default='intrinsic_word_similarity', help='kind of evaluation to perform')
parser.add_argument('-dh', '--data-home', type=str, default='~/.wort_data', help='path to data home')
parser.add_argument('-ds', '--datasets', nargs='+', type=str, help='datasets to use for evaluation', default=[
	'ws353', 'ws353_similarity', 'ws353_relatedness', 'mturk', 'men', 'simlex999'
])


def intrinsic_word_similarity_evaluation(wort_model, datasets=['ws353', 'ws353_similarity', 'ws353_relatedness', 'mturk', 'men', 'simlex999'],
										 distance_fn=cosine, correlation_fn=spearmanr, random_seed=1105, data_home='~/.wort_data', **ds_fetcher_kwargs):
	if (not isinstance(wort_model, VSMVectorizer)):
		wort_model = VSMVectorizer.load_from_file(wort_model)

	logging.info('Evaluating model on {} datasets[type={}]: {}...'.format(len(datasets), type(datasets), datasets))

	if (isinstance(datasets, str) or (len(datasets) == 1 and ',' in datasets[0])):
		datasets = datasets[0].split(',')

	results = {}

	for ds_key in datasets:
		logging.info('Evaluating model on {}...'.format(ds_key))
		ds = DATASET_FETCH_MAP[ds_key](data_home=data_home, **ds_fetcher_kwargs)

		scores = []
		human_sims = []
		for w1, w2, sim in ds:
			if (w1 not in wort_model or w2 not in wort_model):
				logging.warning('"{}" or "{}" not in model vocab! Assigning sim_score=0'.format(w1, w2))
				human_sims.append(sim)
				scores.append(0)
			else:
				human_sims.append(sim)
				scores.append(1 - distance_fn(wort_model[w1].A, wort_model[w2].A))

		model_performance = correlation_fn(np.array(human_sims), np.array(scores))

		logging.info('[{}] - score: {}!'.format(ds_key, model_performance))
		results[ds_key] = model_performance

	return results


# TODO: 3cosmul, 3cosadd (https://www.cs.bgu.ac.il/~yoavg/publications/conll2014analogies.pdf), standard, etc
def intrinsic_word_analogy_evaluation(wort_model, ds_fetcher, distance_fn=cosine, strategy='standard', random_seed=1105, num_neighbours=5, **ds_fetcher_kwargs):
	raise NotImplementedError # This shouldnt be used yet

	# strategy can be 'standard', '3cosmul' or '3cosadd'
	if (not isinstance(wort_model, VSMVectorizer)):
		wort_model = VSMVectorizer.load_from_file(wort_model)

	ds = ds_fetcher(**ds_fetcher_kwargs)

	random.seed(random_seed)

	neighbours = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=num_neighbours).fit(wort_model.get_matrix())
	wort_idx = wort_model.get_index()

	correct = []

	for w1, w2, w3, a in tqdm(ds):
		if (w1 not in wort_model or w2 not in wort_model or w3 not in wort_model):
			wort = wort_idx[random.randint(0, len(wort_idx)-1)]
			logging.warning('"{}" or "{}" or "{}" not in model vocab! Assigning random word="{}"'.format(w1, w2, w3, wort))
			correct.append(wort == a)
		else:
			# TODO: Vectorize the evaluation bit, otherwise it takes an eternity
			v1 = wort_model[w1]
			v2 = wort_model[w2]
			v3 = wort_model[w3]

			# TODO: support the other `strategies` here
			n = v2 - (v1 + v3)
			idx = neighbours.kneighbors(n, return_distance=False)

			for i in idx.squeeze():
				wort = wort_idx[i]
				if (wort != w1 and wort != w2 and wort != w3): # Exclude the query words
					correct.append(wort == a)
					break

	# False=0; True=1
	counts = np.bincount(correct)

	accuracy = (counts / counts.sum())[1]

	return accuracy


if (__name__ == '__main__'):
	args = parser.parse_args()

	if (args.evaluation == 'intrinsic_word_similarity'):
		wort_model = os.path.join(args.input_path, args.input_file)
		intrinsic_word_similarity_evaluation(wort_model=wort_model, datasets=args.datasets, data_home=args.data_home)
	else:
		raise ValueError('Unknown evaluation: {}!'.format(args.evaluation))
