__author__ = 'thomas'
from io import BytesIO
from urllib.request import urlopen
import bz2
import os
import tarfile


def download_wikipedia_dump(data_home='~/.wort_data/wikipedia', url='https://dumps.wikimedia.org/enwiki', dump='latest',
							filename='enwiki-latest-pages-articles.xml.bz2', page_views_url='https://dumps.wikimedia.org'
																							'/other/pagecounts-raw/2016/'
																							'2016-08',
							page_view_file='pagecounts-20160805-120000.gz'):
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

	if (page_views_url is not None):
		print('Downloading page view statistics...')
		with urlopen('{}/{}'.format(page_views_url, page_view_file)) as page_view_dump:
			meta = page_view_dump.info()
			print('Downloading data from {} ({} mb)'.format(page_views_url, round(int(meta['Content-Length'])/1000000)))
			with tarfile.open(os.path.join(data_home, page_view_file), 'r:gz', BytesIO(page_view_dump.read())) as tar:
				tar.extractall(path=os.path.join(data_home))
		print('Download finished!')


def extract_wikipedia_dump():
	# TODO: check for extractor: http://medialab.di.unipi.it/wiki/Wikipedia_Extractor
	pass


def preprocess_wikipedia_dump(min_page_views=0, top_n_pages='all'):
	pass


if (__name__ == '__main__'):
	download_wikipedia_dump()