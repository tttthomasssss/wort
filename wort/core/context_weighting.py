__author__ = 'thomas'
from scipy.special import expit as sigmoid
from scipy.stats import norm

GAUSSIAN = norm(0, 1)
NORM_0 = GAUSSIAN.pdf(0)


def constant_window_weighting(*_):
	return 1


def aggressive_window_weighting(distance, _):
	return 2 ** (1 - abs(distance))


def very_aggressive_window_weighting(distance, _):
	return 2 ** (1 - (distance ** 2))


def harmonic_window_weighting(distance, _):
	return 1. / abs(distance) # Thats what GloVe is doing


def distance_window_weighting(distance, window_size):
	return max(window_size - abs(distance), 1) / window_size # Thats what word2vec is doing


def sigmoid_window_weighting(distance, _):
	return sigmoid(distance)


def inverse_sigmoid_window_weighting(distance, _):
	return 1 - sigmoid(distance)


def absolute_sigmoid_window_weighting(distance, _):
	return abs(sigmoid(distance))


def absolut_inverse_sigmoid_window_weighting(distance, _):
	return 1 - abs(sigmoid(distance))


def inverse_harmonic_window_weighting(distance, _):
	return abs(distance)


def gaussian_window_weighting(distance, _):
	return GAUSSIAN.pdf(abs(distance)-1) / NORM_0