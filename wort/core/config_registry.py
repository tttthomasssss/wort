__author__ = 'thomas'
import os
import sqlite3
import uuid


class ConfigRegistry(object):
	def __init__(self, path, min_frequency, lowercase, stop_words, encoding, max_features, preprocessor, tokenizer,
				 analyzer, token_pattern, decode_error, strip_accents, input, ngram_range, random_state, subsampling_rate,
				 wort_white_list, window_size, context_window_weighting, binary, weighting, cds, sppmi_shift):
		self.db_path_ = os.path.join(path, 'wort_config_registry.sqlite')

		self.min_frequency_ = min_frequency
		self.lowercase_ = lowercase
		self.stop_words_ = stop_words
		self.encoding_ = encoding
		self.max_features_ = max_features
		self.preprocessor_ = preprocessor
		self.tokenizer_ = tokenizer
		self.analyzer_ = analyzer
		self.token_pattern_ = token_pattern
		self.decode_error_ = decode_error
		self.strip_accents_ = strip_accents
		self.input_ = input
		self.ngram_range_ = ngram_range
		self.random_state_ = random_state
		self.subsampling_rate_ = subsampling_rate
		self.wort_white_list_ = wort_white_list
		self.window_size_ = window_size
		self.context_window_weighting_ = context_window_weighting
		self.binary_ = binary
		self.weighting_ = weighting
		self.cds_ = cds
		self.sppmi_shift_ = sppmi_shift

		self._setup()

	def _setup(self):
		if (not os.path.exists(self.db_path_)):
			conn = sqlite3.connect(self.db_path_)
			cursor = conn.cursor()

			vocab_table = """
				CREATE TABLE IF NOT EXISTS Vocab (
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
				);
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
					window_size TEXT,
					context_window_weighting TEXT,
					binary INTEGER,
					sub_folder TEXT
				);
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
					window_size TEXT,
					context_window_weighting TEXT,
					binary INTEGER,
					weighting TEXT,
					cds FLOAT,
					sppmi_shift INTEGER,
					sub_folder TEXT
				);
			"""

			cursor.execute(vocab_table)
			cursor.execute(cooc_table)
			cursor.execute(pmi_table)

			conn.commit()
			conn.close()

	def vocab_cache_folder(self):
		conn = sqlite3.connect(self.db_path_)
		cursor = conn.cursor()

		stmt = """
			SELECT sub_folder FROM Vocab
			WHERE
				min_frequency = ? AND
				lowercase = ? AND
				stop_words = ? AND
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
				wort_white_list = ?;
		"""

		cursor.execute(stmt, (self.min_frequency_, 1 if self.lowercase_ else 0, str(self.stop_words_), self.encoding_,
							  -1 if self.max_features_ is None else self.max_features_, str(self.preprocessor_),
							  str(self.tokenizer_), str(self.analyzer_), self.token_pattern_, self.decode_error_,
							  str(self.strip_accents_),  self.input_, str(self.ngram_range_), str(self.random_state_),
							  0.0 if self.subsampling_rate_ is None else self.subsampling_rate_,
							  str(self.wort_white_list_)))

		return cursor.fetchone()[0]

	def register_vocab(self):
		conn = sqlite3.connect(self.db_path_)
		cursor = conn.cursor()

		stmt = """
			INSERT INTO Vocab (min_frequency, lowercase, encoding, max_features, preprocessor, tokenizer, analyzer,
							token_pattern, decode_error, strip_accents, input, ngram_range, random_state,
							subsampling_rate, wort_white_list, sub_folder)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
		"""

		sub_folder = str(uuid.uuid1())
		cursor.execute(stmt, (self.min_frequency_, 1 if self.lowercase_ else 0, str(self.stop_words_), self.encoding_,
							  -1 if self.max_features_ is None else self.max_features_, str(self.preprocessor_),
							  str(self.tokenizer_), str(self.analyzer_), self.token_pattern_, self.decode_error_,
							  str(self.strip_accents_),  self.input_, str(self.ngram_range_), str(self.random_state_),
							  0.0 if self.subsampling_rate_ is None else self.subsampling_rate_,
							  str(self.wort_white_list_), sub_folder))

		return sub_folder

	def cooccurrence_matrix_folder(self):
		conn = sqlite3.connect(self.db_path_)
		cursor = conn.cursor()

		stmt = """
			SELECT sub_folder FROM Vocab
			WHERE
				min_frequency = ? AND
				lowercase = ? AND
				stop_words = ? AND
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
				window_size = ? AND,
				context_window_weighting = ? AND,
				binary = ?;
		"""

		cursor.execute(stmt, (self.min_frequency_, 1 if self.lowercase_ else 0, str(self.stop_words_), self.encoding_,
							  -1 if self.max_features_ is None else self.max_features_, str(self.preprocessor_),
							  str(self.tokenizer_), str(self.analyzer_), self.token_pattern_, self.decode_error_,
							  str(self.strip_accents_),  self.input_, str(self.ngram_range_), str(self.random_state_),
							  0.0 if self.subsampling_rate_ is None else self.subsampling_rate_,
							  str(self.wort_white_list_), str(self.window_size_), self.context_window_weighting_,
							  1 if self.binary_ else 0))

		return cursor.fetchone()[0]

	def register_cooccurrence_matrix(self):
		conn = sqlite3.connect(self.db_path_)
		cursor = conn.cursor()

		stmt = """
			INSERT INTO Cooccurrence_Matrix (min_frequency, lowercase, encoding, max_features, preprocessor, tokenizer,
							analyzer, token_pattern, decode_error, strip_accents, input, ngram_range, random_state,
							subsampling_rate, wort_white_list, window_size, context_window_weighting, binary, sub_folder)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
		"""

		sub_folder = str(uuid.uuid1())
		cursor.execute(stmt, (self.min_frequency_, 1 if self.lowercase_ else 0, str(self.stop_words_), self.encoding_,
							  -1 if self.max_features_ is None else self.max_features_, str(self.preprocessor_),
							  str(self.tokenizer_), str(self.analyzer_), self.token_pattern_, self.decode_error_,
							  str(self.strip_accents_),  self.input_, str(self.ngram_range_), str(self.random_state_),
							  0.0 if self.subsampling_rate_ is None else self.subsampling_rate_,
							  str(self.wort_white_list_), str(self.window_size_), self.context_window_weighting_,
							  1 if self.binary_ else 0, sub_folder))

		return sub_folder

	def pmi_matrix_folder(self):
		conn = sqlite3.connect(self.db_path_)
		cursor = conn.cursor()

		stmt = """
			SELECT sub_folder FROM Vocab
			WHERE
				min_frequency = ? AND
				lowercase = ? AND
				stop_words = ? AND
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
				window_size = ? AND,
				context_window_weighting = ? AND,
				binary = ? AND
				weighting = ? AND
				cds = ? AND
				sppmi_shift = ?;
		"""

		cursor.execute(stmt, (self.min_frequency_, 1 if self.lowercase_ else 0, str(self.stop_words_), self.encoding_,
							  -1 if self.max_features_ is None else self.max_features_, str(self.preprocessor_),
							  str(self.tokenizer_), str(self.analyzer_), self.token_pattern_, self.decode_error_,
							  str(self.strip_accents_),  self.input_, str(self.ngram_range_), str(self.random_state_),
							  0.0 if self.subsampling_rate_ is None else self.subsampling_rate_,
							  str(self.wort_white_list_), str(self.window_size_), self.context_window_weighting_,
							  1 if self.binary_ else 0, self.weighting_, self.cds_, self.sppmi_shift_))

		return cursor.fetchone()[0]

	def register_pmi_matrix(self):
		conn = sqlite3.connect(self.db_path_)
		cursor = conn.cursor()

		stmt = """
			INSERT INTO PMI_Matrix (min_frequency, lowercase, encoding, max_features, preprocessor, tokenizer,
							analyzer, token_pattern, decode_error, strip_accents, input, ngram_range, random_state,
							subsampling_rate, wort_white_list, window_size, context_window_weighting, binary, weighting,
							cds, sppmi_shift, sub_folder)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
		"""

		sub_folder = str(uuid.uuid1())
		cursor.execute(stmt, (self.min_frequency_, 1 if self.lowercase_ else 0, str(self.stop_words_), self.encoding_,
							  -1 if self.max_features_ is None else self.max_features_, str(self.preprocessor_),
							  str(self.tokenizer_), str(self.analyzer_), self.token_pattern_, self.decode_error_,
							  str(self.strip_accents_),  self.input_, str(self.ngram_range_), str(self.random_state_),
							  0.0 if self.subsampling_rate_ is None else self.subsampling_rate_,
							  str(self.wort_white_list_), str(self.window_size_), self.context_window_weighting_,
							  1 if self.binary_ else 0, self.weighting_, self.cds_, self.sppmi_shift_, sub_folder))

		return sub_folder

