# wort
Am Anfang war das Wort - am Ende die Phrase.

_~~~ Stanislaw Jerzy Lec ~~~_


`wort` is a `python` library for creating count-based distributional semantic word vectors. It adopts a `scikit-learn` like API and is built on top of `numpy`, `scipy` and `scikit-learn`.

# Table of Contents
1. [Installation](Page#installation)

2. [Quickstart](Page#quickstart)
	
3. [Reading a corpus from disk](Page#reading-a-corpus-from-disk)

4. [Creating and fitting `wort` models](Page#creating-and-fitting-wort-models)

5. [Serialising and deserialising `wort` models](Page#serialising-and-deserialising-wort-models)

6. [Accessing individual word vectors](Page#accessing-individual-word-vectors)

7. [Evaluating `wort` models](Page#evaluating-wort-models)

8. [Optimising model throughput](Page#optimising-model-throughput)

---

#### Installation

First

	git clone https://github.com/tttthomasssss/wort.git
	cd wort

Then 
	
	pip install -e .

Or

	python setup.py install

---

#### Quickstart
	
	from wort.corpus_readers import TextStreamReader
	from wort.vsm import VSMVectorizer
	
	# Create PPMI vectors with a symmetric window of 5 from a lowercased corpus, discarding all items occurring less than 100 times
	wort = VSMVectorizer(window_size=5, weighting='ppmi', min_frequency=100, lowercase=True)
	
	corpus_path = 'path/to/corpus/on/disk.txt
	corpus = TextStreamReader(corpus_path)
	
	wort.fit(corpus) # Depending on the size of the corpus, this can take a while...
	
	# Serialise model for later usage
	wort.save_to_file('some/path/to/store/the/model')

---

#### Reading a corpus from disk

Creating meaningful word vector representations requires _a lot_ of data (e.g. all of Wikipedia or all of Project Gutenberg). 

`wort` expects 1 line in the corpus file to correspond to 1 document in the corpus (e.g. 1 Wikipedia article or 1 book from Project Gutenberg).

`wort` provides a few basic corpus readers in `wort.corpus_readers` to deal with corpora in `txt`, `csv`/`tsv` and `gzip` format (assuming 1 line = 1 document).
	
	corpus_path = 'path/to/corpus'
	
	# Reading txt files
	from wort.corpus_readers import TextStreamReader
	
	corpus = TextStreamReader(corpus_path)
	
	# Reading csv/tsv files
	from wort.corpus_readers import CSVStreamReader
	
	corpus = CSVStreamReader(corpus_path, delimiter='\t') # tsv file, the default assues delimiter=',' (csv file)
	
	# Reading gzip files
	from wort.corpus_readers import GzipStreamReader(corpus_path)
	
	corpus = GzipStreamReader(corpus_path)
	
Any of the `corpus` objects can then be passed to the `fit()` method
	
	from wort.vsm import VSMVectorizer
	
	wort = VSMVectorizer(...)
	
	wort.fit(corpus)

`wort` requires two passes over the corpus, the first pass extracts the vocabulary and the second pass constructs the count co-occurrence matrix given the vocabulary.

---

#### Creating and fitting `wort` models

So far, `wort` offers a range of `ppmi` based parameterisations (in addition to some common `scikit-learn` `Vectorizer` options):

* `weighting`: `wort` currently supports `weighting='ppmi'` (with support for `weighting='plmi' and `weighting='pnpmi'` about to be implemented). However, a callable can be passed as well and needs to accept 4 values, the raw PMI matrix (`sparse.csr_matrix`), the matrix of joint probabilities P(w, c) (**!!!ATTENTION: Currently `None` is passed instead of the matrix!!!**), a `numpy.ndarray` vector representing P(w) and a `numpy.ndarray` vector representing P(c).
* `window_size`: Size of the sliding window, accepts symmetric windows (e.g. `window_size=5` or `window_size=(5, 5)`), or asymmetric windows (e.g. `window_size=(1, 5)`)
* `context_window_weighting`: Weighting to the items within the sliding window, default is `context_window_weighting='constant'`, but a range of other schemes are supported (so far `'aggressive'`, `'very_aggressive'`, `'harmonic'` (thats what `GloVe` is doing), `'distance'` (thats what `word2vec` is doing), `'sigmoid'`, `'inverse_sigmoid'`, `'absolute_sigmoid'`, `'inverse_absolute_sigmoid'`). Again, a callable can be passed as well and needs to accept a `distance` parameter, representing the distance from the current word and a `window_size` parameter, representing the size of the window under consideration (Note that this may not be equivalent to the `window_size` parameter used to create the `wort` object). 
* `min_frequency`: Words with a frequency < `min_frequency` will be filtered and discarded
* `binary`: If set to `True`, converts the count based co-occurrence matrix to a binary indicator matrix
* `sppmi_shift`: Subtracts `sppmi_shift` from all non-zero entries of the final PMI matrix. This is equivalent to the number of negative samples in `word2vec`, see [Levy & Goldberg (2014)](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf) and [Levy et al. (2015)](http://www.aclweb.org/anthology/Q/Q15/Q15-1016.pdf) for more information.
* `cds`: Context distribution smoothing, performs `p(c) ** cds`, typically it was found that `cds=0.75` performs particularly well, again see [Levy et al. (2015)](http://www.aclweb.org/anthology/Q/Q15/Q15-1016.pdf) for more information.
* `dim_reduction`: Perform dimensionality reduction on the PMI matrix, currently only `dim_reduction='svd'` is supported.
* `svd_dim`: Dimensionality of the reduced space.
* `svd_eig_weighting`: Eigenvalue weighting of the SVD reduced space, [Levy et al. (2015)](http://www.aclweb.org/anthology/Q/Q15/Q15-1016.pdf) found that `svd_eig_weighting=0.5` or `svd_eig_weighting=0.0` perform better than using SVD the "correct" way.
* `add_context_vectors`: After reducing the dimensionality, the word and context vectors can be added together, see [Pennington et al. (2014)](http://www.aclweb.org/anthology/D/D14/D14-1162.pdf) and [Levy et al. (2015)](http://www.aclweb.org/anthology/Q/Q15/Q15-1016.pdf) for more information.
* `word_white_list`: In academic settings one often evaluates the quality of word vectors on some word similarity dataset. The words in those datasets should obviously not be discarded by a `min_frequency` filter, thus `wort` allows the usage of a white list of words that should not be discarded under any circumstances.

With all of these options in mind, creating `wort` object is as simple as creating a `CountVectorizer` or a `TfidfVectorizer` in `scikit-learn`:

	from wort.vsm import VSMVectorizer
	
	wort = VSMVectorizer(window_size=(1, 7), weighting='ppmi', context_window_weighting='harmonic', min_frequency=100, cds=0.75)
	wort.fit(...)

---

#### Serialising and deserialising `wort` models
	
Given that fitting a distributional model takes a significant amount of time, it is feasible (that means necessary!) to save the models to disk after they've been fitted:

	from wort.vsm import VSMVectorizer
	
	wort = VSMVectorizer(...)
	wort.fit(...)
	
	# Save model to disk
	wort.save_to_file(path='path/to/some/location')
	
The function `save_to_file()` stores the most important assets (not all, to reduce disk space usage) to disk, which includes the final PMI matrix, an index file mapping numbers to words, an inverted index performing the opposite mapping and the word probability distribution P(w).

Once a number of different `wort` models have been created and serialised, loading an existing model is equally simple:

	from wort.vsm import VSMVectorizer
	
	# Load model from disk
	wort = VSMVectorizer.load_from_file(path='path/to/existing/wort/model')

---

#### Accessing individual word vectors

Accessing word vectors adopts a `dict` style aproach:
	
	from wort.vsm import VSMVectorizer
	
	# Load wort model from disk
	wort = VSMVectorizer.load_from_file(path='path/to/existing/wort/model')
	
	v_book = wort['book']

The vector for book is a `1 x N scipy.sparse.csr_matrix`, where `N` is the dimensionality of the vector space, which can be queried by:

	wort.get_vector_size() # Returns an integer
	
Checking whether a word is present in the model can be done by:

	'book' in wort # Returns True or False

---

#### Evaluating `wort` models

	TBD

---

#### Optimising model throughput

	TBD