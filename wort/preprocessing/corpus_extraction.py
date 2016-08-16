__author__ = 'thomas'
from bz2 import decompress
from urllib.parse import urljoin
from urllib.request import urlopen
import bz2
import os


def download_wikipedia_dump(data_home='~/.wort_data/wikipedia', url='https://dumps.wikimedia.org/enwiki', dump='latest',
							filename='enwiki-latest-pages-articles.xml.bz2'):
	data_home = os.path.expanduser(data_home) if '~' in data_home else data_home
	if (not os.path.exists(data_home)):
		os.makedirs(data_home)
	# TODO: Also download page view statistics

	dump_url = '{}/{}/{}'.format(url, dump, filename)
	with urlopen(dump_url) as wikipedia_dump:
		meta = wikipedia_dump.info()
		print('Downloading data from {} ({} mb)'.format(dump_url, round(int(meta['Content-Length'])/1000000)))

		out_file_name = filename.rsplit('.', 1)[0]

		with bz2.open(os.path.join(data_home, out_file_name), 'w') as wiki_dump:
			decompress(wikipedia_dump.read())


def extract_wikipedia_dump():
	# TODO: check for extractor: http://medialab.di.unipi.it/wiki/Wikipedia_Extractor
	pass


def preprocess_wikipedia_dump(min_page_views=0, top_n_pages='all'):
	pass


if (__name__ == '__main__'):
	download_wikipedia_dump()