import os

from common import paths

from wort.corpus_readers import CSVStreamReader
from wort.lm import NGramLanguageModel


def run_lm_pizza_small():
	base_path = os.path.join(paths.get_dataset_path(), 'pizza_small', 'pizza_small.txt')
	f = CSVStreamReader(base_path)

	lm = NGramLanguageModel(token_pattern=r'(?u)\b\w+\b', cache_intermediary_results=False)
	lm.fit(f)

if (__name__ == '__main__'):
	run_lm_pizza_small()