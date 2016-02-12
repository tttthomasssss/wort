__author__ = 'thomas'
import os
import urllib


def get_rubinstein_goodenough_65_words(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'en_rg_65_words.txt'))):
		ds = fetch_rubinstein_goodenough_65_dataset(data_home=data_home)

		words = set()
		for w1, w2, _ in ds:
			words.add(w1)
			words.add(w2)

		with open(os.path.join(data_home, 'en_rg_65_words.txt'), 'w') as word_file:
			for w in words:
				word_file.write(w + '\n')
	else:
		with open(os.path.join(data_home, 'en_rg_65_words.txt'), 'r') as word_file:
			words = set(word_file.read().split('\n'))

	return words


def get_miller_charles_30_words(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'en_mc_30_words.txt'))):
		ds = fetch_miller_charles_30_dataset(data_home=data_home)

		words = set()
		for w1, w2, _ in ds:
			words.add(w1)
			words.add(w2)

		with open(os.path.join(data_home, 'en_mc_30_words.txt'), 'w') as word_file:
			for w in words:
				word_file.write(w + '\n')
	else:
		with open(os.path.join(data_home, 'en_mc_30_words.txt'), 'r') as word_file:
			words = set(word_file.read().split('\n'))

	return words


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


def fetch_miller_charles_30_dataset(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'en_mc_30.txt'))):
		url = 'http://www.cs.cmu.edu/~mfaruqui/word-sim/EN-MC-30.txt'

		if (not os.path.exists(data_home)):
			os.makedirs(data_home)

		with urllib.request.urlopen(url) as rg65:
			data = rg65.read().decode('utf-8')

		with open(os.path.join(data_home, 'en_mc_30.txt'), 'w') as f_out:
			f_out.write(data)
	else:
		with open(os.path.join(data_home, 'en_mc_30.txt'), 'r') as f_in:
			data = f_in.read()

	lines = data.split('\n')
	ds = []
	for line in lines:
		parts = line.strip().split('\t')
		ds.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))

	return ds