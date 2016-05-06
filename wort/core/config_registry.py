__author__ = 'thomas'
import os
import sqlite3
import uuid


class ConfigRegistry(object):
	def __init__(self, path):
		self.db_path_ = os.path.join(path, 'wort_config_registry.sqlite')

		self._setup()

	def _setup(self):
		if (not os.path.exists(self.db_path_)):
			conn = sqlite3.connect(self.db_path_)
			cursor = conn.cursor()

			vocab_table = """
				CRETE TABLE IF NOT EXISTS Vocab (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					min_frequency INTEGER,
					lowercase INTEGER,
					stop_words TEXT,
					encoding TEXT,
					max_features INTEGER,
					preprocessor TEXT,
					tokenizer TEXT,
					analyzer TEXT,
					token_pattern TEXT,
					decode_error TEXT,
					strip_accents TEXT,
					input TEXT,
					ngram_range TEXT,
					random_state TEXT,
					subsampling_rate FLOAT,
					wort_white_list TEXT,
					sub_folder TEXT
				)
			"""

			cooc_table = """
				CREATE TABLE IF NOT EXISTS Cooccurrence_Matrix (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					min_frequency INTEGER,
					lowercase INTEGER,
					stop_words TEXT,
					encoding TEXT,
					max_features INTEGER,
					preprocessor TEXT,
					tokenizer TEXT,
					analyzer TEXT,
					token_pattern TEXT,
					decode_error TEXT,
					strip_accents TEXT,
					input TEXT,
					ngram_range TEXT,
					random_state TEXT,
					subsampling_rate FLOAT,
					wort_white_list TEXT,
					window_size INTEGER,
					context_window_weighting TEXT,
					binary INTEGER,
					sub_folder TEXT
				)
			"""

			pmi_table = """
				CREATE TABLE IF NOT EXISTS PMI_Matrix (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					min_frequency INTEGER,
					lowercase INTEGER,
					stop_words TEXT,
					encoding TEXT,
					max_features INTEGER,
					preprocessor TEXT,
					tokenizer TEXT,
					analyzer TEXT,
					token_pattern TEXT,
					decode_error TEXT,
					strip_accents TEXT,
					input TEXT,
					ngram_range TEXT,
					random_state TEXT,
					subsampling_rate FLOAT,
					wort_white_list TEXT,
					window_size INTEGER,
					context_window_weighting TEXT,
					binary INTEGER,
					weighting TEXT,
					cds FLOAT,
					sppmi_shift INTEGER,
					sub_folder TEXT
				)
			"""

			cursor.execute(vocab_table)
			cursor.execute(cooc_table)
			cursor.execute(pmi_table)

			conn.commit()
			conn.close()

	def vocab_cache_folder(self, min_frequency, lowercase, stop_words, encoding, max_features, preprocessor, tokenizer,
						   analyzer, token_pattern, decode_error, strip_accents, input, ngram_range, random_state,
						   subsampling_rate, wort_white_list):
		conn = sqlite3.connect(self.db_path_)
		cursor = conn.cursor()

		stmt = """
			SELECT sub_folder FROM Vocab
			WHERE
				min_frequency = ? AND
				lowercase = ? AND
				encoding = ? AND
				max_features = ? AND
				preprocessor = ? AND
				tokenizer = ? AND
				analyzer = ? AND
				token_pattern = ? AND
				decode_error = ? AND
				strip_accents = ? AND
				input = ? AND
				ngram_range = ? AND
				random_state = ? AND
				subsampling_rate = ? AND
				wort_white_list = ? AND
		"""

		cursor.execute(stmt, (min_frequency, 1 if lowercase else 0, str(stop_words), encoding,
							  -1 if max_features is None else max_features, str(preprocessor), str(tokenizer),
							  str(analyzer), token_pattern, decode_error, str(strip_accents), input, str(ngram_range),
							  str(random_state), 0.0 if subsampling_rate is None else subsampling_rate,
							  str(wort_white_list)))

		return cursor.fetchone()[0]

	def register_vocab(self, min_frequency, lowercase, stop_words, encoding, max_features, preprocessor, tokenizer,
					   analyzer, token_pattern, decode_error, strip_accents, input, ngram_range, random_state,
					   subsampling_rate, wort_white_list):
		conn = sqlite3.connect(self.db_path_)
		cursor = conn.cursor()

		stmt = """
			INSERT INTO Vocab (min_frequency, lowercase, encoding, max_features, preprocessor, tokenizer, analyzer,
							token_pattern, decode_error, strip_accents, input, ngram_range, random_state,
							subsampling_rate, wort_white_list, sub_folder)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		"""

		sub_folder = str(uuid.uuid1())
		cursor.execute(stmt, (min_frequency, 1 if lowercase else 0, str(stop_words), encoding,
							  -1 if max_features is None else max_features, str(preprocessor), str(tokenizer),
							  str(analyzer), token_pattern, decode_error, str(strip_accents), input, str(ngram_range),
							  str(random_state), 0.0 if subsampling_rate is None else subsampling_rate,
							  str(wort_white_list), sub_folder))

		return sub_folder

