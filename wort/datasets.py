__author__ = 'thomas'
import os
import urllib


def fetch_ws353_dataset(data_home='~/.wort_data'):
	url = 'http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip'

	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(data_home)):
		os.makedirs(data_home)


def fetch_rubinstein_goodenough_65_dataset(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'en_rg_65.txt'))):
		url = 'http://www.cs.cmu.edu/~mfaruqui/word-sim/EN-RG-65.txt'

		if (not os.path.exists(data_home)):
			os.makedirs(data_home)

		with urllib.request.urlopen(url) as rg65:
			data = rg65.read().decode('utf-8')

		with open(os.path.join(data_home, 'en_rg_65.txt'), 'w') as f_out:
			f_out.write(data)
	else:
		with open(os.path.join(data_home, 'en_rg_65.txt'), 'r') as f_in:
			data = f_in.read()

	lines = data.split('\n')
	ds = []
	for line in lines:
		parts = line.strip().split('\t')
		ds.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))

	return ds