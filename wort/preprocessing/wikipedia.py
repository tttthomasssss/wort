__author__ = 'thomas'
from urllib.request import urlopen
import bz2
import gzip
import os
import subprocess


def download_page_view_statistics(data_home='~/.wort_data/wikipedia', url='https://dumps.wikimedia.org'
																		  '/other/pagecounts-raw/2016/'
																		  '2016-08',
								  filename='pagecounts-20160805-120000.gz'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home
	if (not os.path.exists(data_home)):
		os.makedirs(data_home)

	dump_url = '{}/{}'.format(url, filename)
	print('Downloading page view statistics...')
	with urlopen(dump_url) as page_view_dump:
		meta = page_view_dump.info()
		print('Downloading data from {} ({} mb)'.format(dump_url, round(int(meta['Content-Length']) / 1000000)))
		with gzip.open(os.path.join(data_home, filename), 'wb') as gz:
			gz.write(page_view_dump.read())
	print('Download finished!')


def download_corpus_dump(data_home='~/.wort_data/wikipedia', url='https://dumps.wikimedia.org/enwiki', dump='latest',
						 filename='enwiki-latest-pages-articles.xml.bz2'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home
	if (not os.path.exists(data_home)):
		os.makedirs(data_home)

	dump_url = '{}/{}/{}'.format(url, dump, filename)
	with urlopen(dump_url) as wikipedia_dump:
		meta = wikipedia_dump.info()
		print('Downloading data from {} ({} mb)'.format(dump_url, round(int(meta['Content-Length'])/1000000)))

		with bz2.open(os.path.join(data_home, filename), 'w') as wiki_dump:
			wiki_dump.write(wikipedia_dump.read())
	print('Download finished!')


def extract_corpus_dump(data_home='~/.wort_data/wikipedia'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home
	if (not os.path.exists(data_home)):
		os.makedirs(data_home)

	if (not os.path.exists(os.path.join(data_home, 'wikiextractor'))):
		print('WikiExtractor not found, cloning from https://github.com/attardi/wikiextractor.git...')
		process = subprocess.Popen('git clone https://github.com/attardi/wikiextractor.git {}'.format(data_home),
								   stdout=subprocess.PIPE)
		output, error = process.communicate()
		print('OUTPUT={}; ERROR={}'.format(output, error))


def preprocess_corpus_dump(min_page_views=0, top_n_pages='all'):
	pass


if (__name__ == '__main__'):
	download_corpus_dump(data_home='/lustre/scratch/inf/thk22/_datasets/wikipedia/corpus/')
	extract_corpus_dump()