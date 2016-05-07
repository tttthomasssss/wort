__author__ = 'thomas'
import numpy as np


def ppmi_transformation(PMI, *_):
	return PMI


def plmi_transformation(PMI, P_w_c, *_): # TODO: Needs testing (pizza_small.txt)
	return P_w_c.multiply(PMI)


def pnpmi_transformation(PMI, P_w_c, *_): #TODO: Needs testing (pizza_small.txt)
	# Tricky one, could normalise by -log(P(w)), -log(P(c)) or -log(P(w, c)); choose the latter because it normalises the upper & the lower bound,
	# and is nicer implementationwise (see Bouma 2009: https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf)
	P_w_c.data = 1 / -np.log(P_w_c.data)
	return P_w_c.multiply(PMI)