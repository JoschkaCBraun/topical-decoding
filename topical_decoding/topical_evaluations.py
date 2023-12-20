import string
from multiprocessing import Manager, Process
from typing import Optional

import nltk
import spacy
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer

# hyperparameters
num_words = 100  # Number of words to extract from each topic
minimum_phi_value = (
    0.001  # Minimum value of phi to consider when extracting topic words
)


def initialize_nlp_resources():
    """Initializes necessary NLP resources from NLTK and SpaCy."""
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


nlp = initialize_nlp_resources()
stopwords = set(nltk_stopwords.words("english"))
punctuation = set(string.punctuation)
stemmer = PorterStemmer()


def batch_process_texts(texts: list[str], method: str) -> list[set]:
    """
    Processes a batch of texts based on the specified method ('lemmatize' or 'stem').

    :param texts: A list of strings to be processed.
    :param method: The method of text processing ('lemmatize' or 'stem').
    :return: A list of sets of processed words.
    """
    if method == "lemmatize":
        return [
            set(
                token.lemma_
                for token in nlp(text.lower())
                if token.text not in punctuation and token.text not in stopwords
            )
            for text in texts
        ]
    elif method == "stem":
        return [
            {
                stemmer.stem(token)
                for token in nltk.word_tokenize(text.lower())
                if token not in punctuation and token not in stopwords
            }
            for text in texts
        ]
    else:
        raise ValueError("Invalid method. Choose either 'lemmatize' or 'stem'.")


def extract_topic_words_with_weights(lda: LdaModel, tid: int) -> dict[str, float]:
    """
    Extracts words and their corresponding weights for a given topic.

    :param lda: A trained gensim LdaModel.
    :param tid: Topic ID for which words are extracted.
    :return: A dictionary of words and their weights.
    """
    return {word: weight for word, weight in lda.show_topic(tid, topn=num_words)}


def topic_scores(
    documents: list[str],
    lda: LdaModel,
    tids: list[int],
    method: str,
    dictionary: Optional[dict] = None,
) -> list[float]:
    """
    Calculates topic scores for a list of documents.

    :param documents: A list of documents (strings).
    :param lda: A trained gensim LdaModel.
    :param tids: A list of topic IDs.
    :param method: Method for text processing ('dict', 'lemmatize', or 'stem').
    :param dictionary: An optional dictionary for 'dict' method.
    :return: A list of topic scores.
    """
    if not isinstance(documents, list) or not all(
        isinstance(doc, str) for doc in documents
    ):
        raise TypeError("documents must be a list of strings.")
    if not isinstance(lda, LdaModel):
        raise TypeError("lda must be an instance of gensim.models.ldamodel.LdaModel.")
    if not isinstance(tids, list) or not all(isinstance(tid, int) for tid in tids):
        raise TypeError("tids must be a list of integers.")
    if method not in ["dict", "lemmatize", "stem"]:
        raise ValueError("Invalid method. Choose 'dict', 'lemmatize', or 'stem'.")

    if method == "dict":
        if dictionary is None:
            raise ValueError("A dictionary must be provided for the 'dict' method.")
        return batch_dict_topic_score(documents, lda, tids, dictionary)
    else:
        processed_docs = batch_process_texts(documents, method)
        scores = []
        for doc, tid in zip(processed_docs, tids):
            topic_words_with_weights = extract_topic_words_with_weights(lda, tid)
            total_weight = sum(topic_words_with_weights.values())
            if total_weight == 0:
                scores.append(0)
                continue
            weighted_score = sum(topic_words_with_weights.get(word, 0) for word in doc)
            scores.append(weighted_score / total_weight)
        return scores


def batch_dict_topic_score(
    documents: list[str], lda: LdaModel, tids: list[int], dictionary: dict
) -> list[float]:
    """
    Calculates topic scores using a dictionary method for a list of documents, each against a corresponding topic ID.

    :param documents: A list of documents.
    :param lda: A trained LdaModel.
    :param tids: A list of topic IDs, one for each document.
    :param dictionary: A dictionary for text processing.
    :return: A list of topic scores, one for each document.
    """
    if len(documents) != len(tids):
        raise ValueError("The length of documents and tids must be the same.")

    vec_bows = [dictionary.doc2bow(doc.lower().split(" ")) for doc in documents]
    doc_topics = [dict(lda.get_document_topics(vec_bow)) for vec_bow in vec_bows]
    return [prevalences.get(tid, 0.0) for prevalences, tid in zip(doc_topics, tids)]


# Multiprocessing
def process_documents_chunk(doc_chunk, lda, tids_chunk, method, result_list):
    """
    Processes a chunk of documents and appends the calculated topic scores to a shared list.

    :param doc_chunk: A subset of documents.
    :param lda: A trained gensim LdaModel.
    :param tids_chunk: A subset of topic IDs corresponding to the documents.
    :param method: Method for text processing ('lemmatize' or 'stem').
    :param result_list: A shared list to store the results.
    """
    processed_docs = batch_process_texts(doc_chunk, method)
    local_scores = []

    for doc, tid in zip(processed_docs, tids_chunk):
        topic_words_with_weights = extract_topic_words_with_weights(lda, tid)
        total_weight = sum(topic_words_with_weights.values())

        if total_weight == 0:
            local_scores.append(0)
            continue

        weighted_score = sum(topic_words_with_weights.get(word, 0) for word in doc)
        local_scores.append(weighted_score / total_weight)

    result_list.extend(local_scores)


def topic_score_multiprocess(documents, lda, tids, method, num_processes=4):
    """
    Calculates topic scores for a list of documents using multiple processes.

    :param documents: A list of documents (strings).
    :param lda: A trained gensim LdaModel.
    :param tids: A list of topic IDs.
    :param method: Method for text processing ('lemmatize' or 'stem').
    :param num_processes: Number of processes to use (default is 4).
    :return: A list of topic scores.
    """
    if len(documents) != len(tids):
        raise ValueError("The length of documents and tids must be the same.")

    manager = Manager()
    result_list = manager.list()
    processes = []
    chunk_size = len(documents) // num_processes + (len(documents) % num_processes > 0)

    for i in range(num_processes):
        start = i * chunk_size
        end = None if i == num_processes - 1 else start + chunk_size
        doc_chunk = documents[start:end]
        tids_chunk = tids[start:end]
        process = Process(
            target=process_documents_chunk,
            args=(doc_chunk, lda, tids_chunk, method, result_list),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    return list(result_list)
