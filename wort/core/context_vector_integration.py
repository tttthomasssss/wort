__author__ = 'thomas'


def add_context_vectors(W, C, w_weights=1., c_weights=1.):
	return (W * w_weights) + (C * c_weights) if w_weights != 1. and c_weights != 1. else W + C


def multiply_context_vectors(W, C, w_weights=1., c_weights=1.):
	return (W * w_weights) * (C * c_weights) if w_weights != 1. and c_weights != 1. else W * C