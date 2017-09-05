__author__ = 'thomas'
from io import BytesIO
from zipfile import ZipFile
import functools
import os
import tarfile
import urllib

# TODO: Lots of the dataset loaders work the same way --> encapsulate into single function (e.g. WS353 full, MEN, RW, ...)
# TODO: Ditto with the `words` loaders

def get_msr_syntactic_analogies_words(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'MSR_Syntactic_Analogies', 'msr_worts.txt'))):
		ds = fetch_msr_syntactic_analogies_dataset(data_home=data_home)

		words = set()
		for w1, w2, w3, a in ds:
			if (w1 != ''):
				words.add(w1)
			if (w2 != ''):
				words.add(w2)
			if (w3 != ''):
				words.add(w3)
			if (a != ''):
				words.add(a)

		with open(os.path.join(data_home, 'MSR_Syntactic_Analogies', 'msr_worts.txt'), 'w') as word_file:
			for w in words:
				word_file.write(w + '\n')
	else:
		with open(os.path.join(data_home, 'MSR_Syntactic_Analogies', 'msr_worts.txt'), 'r') as word_file:
			words = set(word_file.read().split('\n'))

	return words


def get_google_analogies_words(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'word2vec', 'goog_worts.txt'))):
		ds = fetch_google_analogies_dataset(data_home=data_home)

		words = set()
		for w1, w2, w3, a in ds:
			if (w1 != ''):
				words.add(w1)
			if (w2 != ''):
				words.add(w2)
			if (w3 != ''):
				words.add(w3)
			if (a != ''):
				words.add(a)

		with open(os.path.join(data_home, 'word2vec', 'goog_worts.txt'), 'w') as word_file:
			for w in words:
				word_file.write(w + '\n')
	else:
		with open(os.path.join(data_home, 'word2vec', 'goog_worts.txt'), 'r') as word_file:
			words = set(word_file.read().split('\n'))

	return words


def get_simlex_999_words(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'SimLex-999', 'simlex_worts.txt'))):
		ds = fetch_rare_words_dataset(data_home=data_home)

		words = set()
		for w1, w2, _ in ds:
			if (w1 != ''):
				words.add(w1)
			if (w2 != ''):
				words.add(w2)

		with open(os.path.join(data_home, 'SimLex-999', 'simlex_worts.txt'), 'w') as word_file:
			for w in words:
				word_file.write(w + '\n')
	else:
		with open(os.path.join(data_home, 'SimLex-999', 'simlex_worts.txt'), 'r') as word_file:
			words = set(word_file.read().split('\n'))

	return words


def get_ws353_words(data_home='~/.wort_data', subset='all', similarity_type=None):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (similarity_type == 'similarity' or similarity_type == 'relatedness'):
		if (not os.path.exists(os.path.join(data_home, 'ws353', 'wordsim353_sim_rel', 'ws353_{}_wort.txt'.format(similarity_type)))):
			ds = fetch_ws353_dataset(data_home=data_home, subset=subset, similarity_type=similarity_type)

			words = set()
			for w1, w2, _ in ds:
				if (w1 != ''):
					words.add(w1)
				if (w2 != ''):
					words.add(w2)

			with open(os.path.join(data_home, 'ws353', 'wordsim353_sim_rel', 'ws353_{}_wort.txt'.format(similarity_type)), 'w') as word_file:
				for w in words:
					word_file.write(w + '\n')
		else:
			with open(os.path.join(data_home, 'ws353', 'wordsim353_sim_rel', 'ws353_{}_wort.txt'.format(similarity_type)), 'r') as word_file:
				words = set(word_file.read().split('\n'))
	else:
		if (not os.path.exists(os.path.join(data_home, 'ws353', 'original', 'ws353_{}_wort.txt'.format(subset)))):
			ds = fetch_ws353_dataset(data_home=data_home, subset=subset, similarity_type=similarity_type)

			words = set()
			for w1, w2, _ in ds:
				if (w1 != ''):
					words.add(w1)
				if (w2 != ''):
					words.add(w2)

			with open(os.path.join(data_home, 'ws353', 'original', 'ws353_{}_wort.txt'.format(subset)), 'w') as word_file:
				for w in words:
					word_file.write(w + '\n')
		else:
			with open(os.path.join(data_home, 'ws353', 'original', 'ws353_{}_wort.txt'.format(subset)), 'r') as word_file:
				words = set(word_file.read().split('\n'))

	return words


def get_mturk_words(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'mturk_worts.txt'))):
		ds = fetch_mturk_dataset(data_home=data_home)

		words = set()
		for w1, w2, _ in ds:
			if (w1 != ''):
				words.add(w1)
			if (w2 != ''):
				words.add(w2)

		with open(os.path.join(data_home, 'mturk_worts.txt'), 'w') as word_file:
			for w in words:
				word_file.write(w + '\n')
	else:
		with open(os.path.join(data_home, 'mturk_worts.txt'), 'r') as word_file:
			words = set(word_file.read().split('\n'))

	return words


def get_rare_words(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'rw', 'rare_worts.txt'))):
		ds = fetch_rare_words_dataset(data_home=data_home)

		words = set()
		for w1, w2, _ in ds:
			if (w1 != ''):
				words.add(w1)
			if (w2 != ''):
				words.add(w2)

		with open(os.path.join(data_home, 'rw', 'rare_worts.txt'), 'w') as word_file:
			for w in words:
				word_file.write(w + '\n')
	else:
		with open(os.path.join(data_home, 'rw', 'rare_worts.txt'), 'r') as word_file:
			words = set(word_file.read().split('\n'))

	return words


def get_rubinstein_goodenough_65_words(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'en_rg_65_words.txt'))):
		ds = fetch_rubinstein_goodenough_65_dataset(data_home=data_home)

		words = set()
		for w1, w2, _ in ds:
			if (w1 != ''):
				words.add(w1)
			if (w2 != ''):
				words.add(w2)

		with open(os.path.join(data_home, 'en_rg_65_words.txt'), 'w') as word_file:
			for w in words:
				word_file.write(w + '\n')
	else:
		with open(os.path.join(data_home, 'en_rg_65_words.txt'), 'r') as word_file:
			words = set(word_file.read().split('\n'))

	return words


def get_men_words(data_home='~/.wort_data', lemma=True):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'MEN', 'men_worts.txt'))):
		ds = fetch_men_dataset(data_home=data_home, lemma=lemma)

		words = set()
		for w1, w2, _ in ds:
			if (w1 != ''):
				words.add(w1)
			if (w2 != ''):
				words.add(w2)

		with open(os.path.join(data_home, 'MEN', 'men_worts.txt'), 'w') as word_file:
			for w in words:
				word_file.write(w + '\n')
	else:
		with open(os.path.join(data_home, 'MEN', 'men_worts.txt'), 'r') as word_file:
			words = set(word_file.read().split('\n'))

	return words


def get_miller_charles_30_words(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'en_mc_30_words.txt'))):
		ds = fetch_miller_charles_30_dataset(data_home=data_home)

		words = set()
		for w1, w2, _ in ds:
			if (w1 != ''):
				words.add(w1)
			if (w2 != ''):
				words.add(w2)

		with open(os.path.join(data_home, 'en_mc_30_words.txt'), 'w') as word_file:
			for w in words:
				word_file.write(w + '\n')
	else:
		with open(os.path.join(data_home, 'en_mc_30_words.txt'), 'r') as word_file:
			words = set(word_file.read().split('\n'))

	return words


def fetch_ws353_dataset(data_home='~/.wort_data', subset='all', similarity_type=None):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home
	ds_home = os.path.join(data_home, 'ws353')

	if (not os.path.exists(ds_home)):
		os.makedirs(ds_home)

	if (similarity_type == 'similarity' or similarity_type == 'relatedness'):
		if (not os.path.exists(os.path.join(ds_home, 'wordsim353_sim_rel'))):
			os.makedirs(os.path.join(ds_home, 'wordsim353_sim_rel'))

			url = 'http://alfonseca.org/pubs/ws353simrel.tar.gz'
			with urllib.request.urlopen(url) as ws353:
				meta = ws353.info()
				print('Downloading data from {} ({} kb)'.format(url, round(int(meta['Content-Length'])/1000)))

				with tarfile.open(os.path.join(ds_home, 'simrel', 'ws353simrel.tar.gz'), 'r:gz', BytesIO(ws353.read())) as tar:
					tar.extractall(path=os.path.join(ds_home))

		fname = 'wordsim_similarity_goldstandard.txt' if (similarity_type == 'similarity') else 'wordsim_relatedness_goldstandard.txt'
		fpath = os.path.join(ds_home, 'wordsim353_sim_rel')
		skip_header = False

	else:
		if (not os.path.exists(os.path.join(ds_home, 'original'))):
			os.makedirs(os.path.join(ds_home, 'original'))

			url = 'http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip'
			with urllib.request.urlopen(url) as ws353:
				meta = ws353.info()
				print('Downloading data from {} ({} kb)'.format(url, round(int(meta['Content-Length'])/1000)))

				zip = ZipFile(BytesIO(ws353.read()))
				zip.extractall(path=os.path.join(ds_home, 'original'))
				zip.close()

		fname = '{}.tab'.format(subset) if subset == 'set1' or subset == 'set2' else 'combined.tab'
		fpath = os.path.join(ds_home, 'original')
		skip_header = True

	ds = []
	with open(os.path.join(fpath, fname), 'r') as ws353_file:
		if (skip_header):
			next(ws353_file)

		for line in ws353_file:
			parts = line.strip().lower().split('\t')
			ds.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))

	return ds


def fetch_rubinstein_goodenough_65_dataset(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(os.path.join(data_home, 'en_rg_65.txt'))):
		url = 'http://www.cs.cmu.edu/~mfaruqui/word-sim/EN-RG-65.txt'

		if (not os.path.exists(data_home)):
			os.makedirs(data_home)

		with urllib.request.urlopen(url) as rg65:
			meta = rg65.info()
			print('Downloading data from {} ({} b)'.format(url, int(meta['Content-Length'])))
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

		with urllib.request.urlopen(url) as mc30:
			meta = mc30.info()
			print('Downloading data from {} ({} b)'.format(url, int(meta['Content-Length'])))
			data = mc30.read().decode('utf-8')

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


def fetch_rare_words_dataset(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(data_home)):
		os.makedirs(data_home)

	if (not os.path.exists(os.path.join(data_home, 'rw'))):
		url = 'http://www-nlp.stanford.edu/~lmthang/morphoNLM/rw.zip'

		with urllib.request.urlopen(url) as rw:
			meta = rw.info()
			print('Downloading data from {} ({} kb)'.format(url, round(int(meta['Content-Length'])/1000)))

			zip = ZipFile(BytesIO(rw.read()))
			zip.extractall(path=os.path.join(data_home))
			zip.close()

	with open(os.path.join(data_home, 'rw', 'rw.txt'), 'r') as ds_file:
		ds = []
		for line in ds_file:
			parts = line.lower().strip().split('\t')
			ds.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))

	return ds


def fetch_men_dataset(data_home='~/.wort_data', lemma=True):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(data_home)):
		os.makedirs(data_home)

	form = 'lemma' if lemma else 'natural'

	if (not os.path.exists(os.path.join(data_home, 'MEN'))):
		url = 'https://staff.fnwi.uva.nl/e.bruni/resources/MEN.zip'

		with urllib.request.urlopen(url) as men:
			meta = men.info()
			print('Downloading data from {} ({} kb)'.format(url, round(int(meta['Content-Length'])/1000)))

			zip = ZipFile(BytesIO(men.read()))
			zip.extractall(path=os.path.join(data_home))
			zip.close()

	with open(os.path.join(data_home, 'MEN', 'MEN_dataset_{}_form_full'.format(form)), 'r') as ds_file:
		ds = []
		for line in ds_file:
			parts = line.lower().strip().split()
			ds.append((parts[0].strip().split('-')[0], parts[1].strip().split('-')[0], float(parts[2].strip())))

	return ds


def fetch_mturk_dataset(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(data_home)):
		os.makedirs(data_home)

	if (not os.path.exists(os.path.join(data_home, 'Mtruk.csv'))):
		url = 'http://tx.technion.ac.il/~kirar/files/Mtruk.csv'

		with urllib.request.urlopen(url) as mturk:
			meta = mturk.info()
			print('Downloading data from {} ({} kb)'.format(url, round(int(meta['Content-Length'])/1000)))
			data = mturk.read().decode('utf-8')

		with open(os.path.join(data_home, 'Mtruk.csv'), 'w') as f_out:
			f_out.write(data)
	else:
		with open(os.path.join(data_home, 'Mtruk.csv'), 'r') as f_in:
			data = f_in.read()

	lines = data.strip().split('\n')
	ds = []
	for line in lines:
		parts = line.lower().strip().split(',')
		ds.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))

	return ds


def fetch_simlex_999_dataset(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(data_home)):
		os.makedirs(data_home)

	if (not os.path.exists(os.path.join(data_home, 'SimLex-999'))):
		url = 'http://www.cl.cam.ac.uk/~fh295/SimLex-999.zip'

		with urllib.request.urlopen(url) as simlex:
			meta = simlex.info()
			print('Downloading data from {} ({} kb)'.format(url, round(int(meta['Content-Length'])/1000)))

			zip = ZipFile(BytesIO(simlex.read()))
			zip.extractall(path=data_home)
			zip.close()

	with open(os.path.join(data_home, 'SimLex-999', 'SimLex-999.txt'), 'r') as ds_file:
		next(ds_file) # Skip header
		ds = []
		for line in ds_file:
			parts = line.lower().strip().split('\t')
			ds.append((parts[0].strip(), parts[1].strip(), float(parts[3].strip())))

	return ds


def fetch_google_analogies_dataset(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(data_home)):
		os.makedirs(data_home)

	if (not os.path.exists(os.path.join(data_home, 'word2vec'))):
		url = 'https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip'

		with urllib.request.urlopen(url) as word2vec:
			meta = word2vec.info()
			print('Downloading data from {} ({} kb)'.format(url, round(int(meta['Content-Length'])/1000)))

			zip = ZipFile(BytesIO(word2vec.read()))
			zip.extractall(path=data_home)
			zip.close()

	with open(os.path.join(data_home, 'word2vec', 'trunk', 'questions-words.txt'), 'r') as ds_file:
		ds = []
		for line in ds_file:
			if (not line.startswith(': ')): # This marks a header
				parts = line.lower().strip().split()
				ds.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()))

	return ds


def fetch_msr_syntactic_analogies_dataset(data_home='~/.wort_data'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home

	if (not os.path.exists(data_home)):
		os.makedirs(data_home)

	if (not os.path.exists(os.path.join(data_home, 'MSR_Syntactic_Analogies'))):
		ds_home = os.path.join(data_home, 'MSR_Syntactic_Analogies')
		os.makedirs(ds_home)

		url = 'http://research.microsoft.com/en-us/um/people/gzweig/Pubs/myz_naacl13_test_set.tgz'

		with urllib.request.urlopen(url) as msr:
			meta = msr.info()
			print('Downloading data from {} ({} kb)'.format(url, round(int(meta['Content-Length'])/1000)))

			with tarfile.open(os.path.join(ds_home, 'myz_naacl13_test_set.tgz'), 'r:gz', BytesIO(msr.read())) as tar:
				tar.extractall(path=os.path.join(ds_home))

	# Evaluation Data and Solutions
	with open(os.path.join(data_home, 'MSR_Syntactic_Analogies', 'test_set', 'word_relationship.questions'), 'r') as ds_file, \
		open(os.path.join(data_home, 'MSR_Syntactic_Analogies', 'test_set', 'word_relationship.answers'), 'r') as ds_file_answers:

		ds = []
		for line in zip(ds_file, ds_file_answers):
			parts = line[0].lower().strip().split() + line[1].lower().strip().split()
			ds.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[4].strip()))

	return ds


DATASET_KEYS = [
	#'msr_syntactic_analogies', TODO
	#'google_analogies', TODO
	'ws353',
	'ws353_similarity',
	'ws353_relatedness',
	'mc30',
	'rg65',
	'rw',
	'men',
	'mturk',
	'simlex999'
]

DATASET_FETCH_MAP = {
	'ws353': functools.partial(fetch_ws353_dataset, similarity_type=None),
	'ws353_similarity': functools.partial(fetch_ws353_dataset, similarity_type='similarity'),
	'ws353_relatedness': functools.partial(fetch_ws353_dataset, similarity_type='relatedness'),
	'mc30': fetch_miller_charles_30_dataset,
	'rg65': fetch_rubinstein_goodenough_65_dataset,
	'rw': fetch_rare_words_dataset,
	'men': fetch_men_dataset,
	'mturk': fetch_mturk_dataset,
	'simlex999': fetch_simlex_999_dataset
}