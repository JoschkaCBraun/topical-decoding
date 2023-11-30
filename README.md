# topical-decoding
Decoding algorithms that are able to focus on a given topic when generating text.

### README.md for topical_evaluations.py

---

# Topical Evaluations

This repository contains `topical_evaluations.py`, a Python script designed for analyzing document summaries with a focus on topical relevance. The script includes two main functions: `topic_scores` and `topic_score_multiprocess`. These functions are useful for calculating the relevance of topics in document summaries using a given LDA model.

## Features

- **topic_scores**: Calculates topic relevance scores for a list of documents, considering specified topic IDs.
- **topic_score_multiprocess**: Enhances performance by calculating topic scores using multiple processes.

## How to Use

### topic_scores Function

This function calculates topic scores for a given list of documents. 

#### Parameters:
- `documents`: A list of documents (strings).
- `lda`: A trained gensim LdaModel.
- `tids`: A list of topic IDs for which the relevance is calculated.
- `method`: The text processing method ('dict', 'lemmatize', or 'stem').
- `dictionary` (optional): A dictionary used for the 'dict' method.

#### Usage:

```python
from gensim.models import LdaModel

# Assuming you have a list of documents and a trained LdaModel
documents = ["your document text", ...]
lda_model = LdaModel(...)
topic_ids = [0, 1, 2]  # example topic IDs
scores = topic_scores(documents, lda_model, topic_ids, 'lemmatize')
```

### topic_score_multiprocess Function

This function is designed to compute topic scores using multiprocessing for enhanced performance.

#### Parameters:
- `documents`: A list of documents (strings).
- `lda`: A trained gensim LdaModel.
- `tids`: A list of topic IDs.
- `method`: The text processing method ('lemmatize' or 'stem').
- `num_processes`: The number of processes to use (default is 4).

#### Usage:

```python
from gensim.models import LdaModel

# Assuming you have a list of documents and a trained LdaModel
documents = ["your document text", ...]
lda_model = LdaModel(...)
topic_ids = [0, 1, 2]  # example topic IDs
scores = topic_score_multiprocess(documents, lda_model, topic_ids, 'lemmatize', num_processes=4)
```

---

## Installation

Ensure you have the following dependencies installed:
- NLTK
- SpaCy
- Gensim

You can install them using pip:

```bash
pip install nltk spacy gensim
```
