__author__ = 'thomas'
import logging
import random

from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm import *
import numpy as np

from wort.vsm import VSMVectorizer

# TODO: Selector for datasets
# TODO: OOV mode ('ignore', 'random', 0.5)
# TODO: replace NaNs


def intrinsic_word_similarity_evaluation(wort_model, ds_fetcher, distance_fn=cosine, correlation_fn=spearmanr, random_seed=1105, **ds_fetcher_kwargs):
	if (not isinstance(wort_model, VSMVectorizer)):
		wort_model = VSMVectorizer.load_from_file(wort_model)

	ds = ds_fetcher(ds_fetcher_kwargs)

	random.seed(random_seed)

	scores = []
	human_sims = []
	for w1, w2, sim in ds:
		if (w1 not in wort_model or w2 not in wort_model):
			score = random.random()
			logging.warning('"{}" or "{}" not in model vocab! Assigning random sim_score={}'.format(w1, w2, score))
			human_sims.append(sim)
			scores.append(score)
		else:
			human_sims.append(sim)
			scores.append(1 - distance_fn(wort_model[w1].A, wort_model[w2].A))

	model_performance = correlation_fn(np.array(human_sims), np.array(scores))

	logging.info('Model Performance: {}!'.format(model_performance))

	return model_performance

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