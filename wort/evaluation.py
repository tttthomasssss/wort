__author__ = 'thomas'
import logging
import random

from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import numpy as np

from wort.vsm import VSMVectorizer

# TODO: Selector for datasets
# TODO: OOV mode ('ignore', 'random', 0.5)


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
			logging.warning('{} or {} not in model vocab! Assigning random sim_score={}'.format(w1, w2, score))
			human_sims.append(sim)
			scores.append(score)
		else:
			human_sims.append(sim)
			scores.append(1 - distance_fn(wort_model[w1], wort_model[w2]))

	model_performance = correlation_fn(np.array(human_sims), np.array(scores))

	logging.info('Model Performance: {}!'.format(model_performance))

	return model_performance

# TODO: 3cosmul, 3cosadd, standard, etc
def intrinsic_word_analogy_evaluation():
	pass