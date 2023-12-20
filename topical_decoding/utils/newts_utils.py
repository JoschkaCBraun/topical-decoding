"""Functions for reading and evaluating on the NEWTS dataset, along with the
corresponding LDA model (if desired). 

Defines a number of functions that can be used to evaluate summaries based on
whether they are focused on a given topic or not. 

Requires lda model to be loaded in:
    lda, dictionary = readLDA('path/to/model')
"""
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel


# returns a Pandas DataFrame of the NEWTS testing set
def read_test(path="NEWTS/NEWTS_test_600.csv"):
    out = pd.read_csv(path, encoding="utf-8", index_col=[0])
    assert len(out) == 600
    return out


# returns a Pandas DataFrame of the NEWTS training set
def read_train(path="NEWTS/NEWTS_train_2400.csv"):
    out = pd.read_csv(path, encoding="utf-8", index_col=[0])
    assert len(out) == 2400
    return out


# given a path to model, loads it and its dictionary
def read_lda(path="LDA_250/"):
    path = path.strip("/") + "/"
    lda = LdaModel.load(path + "lda.model", mmap="r")
    print("Loaded the LDA model")
    dictionary = Dictionary.load(path + "dictionary.dic", mmap="r")
    print("Loaded dictionary")
    return lda, dictionary


def doc_topics(document, lda, dictionary):
    """
    Calculate the prevalence of various topics within a given document using an LDA
    model.

    This function returns a dictionary where the keys are topic IDs and the values are
    the prevalence of that topic in the document.

    Parameters
    ----------
    document : str
        A string containing a summary or article for which to calculate topics.
    lda : gensim.models.ldamodel.LdaModel
        A trained LDA model, loaded as per instructions in README.
    dictionary : gensim.corpora.dictionary.Dictionary
        A dictionary object loaded following instructions in README.

    Returns
    -------
    dict
        A dictionary with keys as topic IDs and values as the prevalence of that topic
        in the document.

    Notes
    -----
    The LDA model's minimum_phi_value is set to 0.01, and per_word_topics is set to
    False within this function.

    Examples
    --------
    >>> lda = gensim.models.ldamodel.LdaModel.load('model_path')
    >>> dictionary = gensim.corpora.dictionary.Dictionary.load('dictionary_path')
    >>> doc_topics('Sample text of the document', lda, dictionary)
    {0: 0.5, 1: 0.3, ...}
    """

    lda.minimum_phi_value = 0.001
    lda.per_word_topics = False
    vec_bow = dictionary.doc2bow(document.split(" "))
    temp = lda[vec_bow]
    temp.sort(key=lambda x: x[1], reverse=True)
    return dict(temp)


def topic_score(tid, document, lda, dictionary):
    """
    Calculate the prevalence of a specific topic in a given document.

    This function returns a float representing the probability of the specified topic
    within the document, using a trained LDA model.

    Parameters
    ----------
    tid : int
        The ID of the topic for which the prevalence score is to be calculated.
    document : str
        A string containing a summary or article for which to calculate the topic score.
    lda : gensim.models.ldamodel.LdaModel
        A trained LDA model, loaded as per instructions in README.
    dictionary : gensim.corpora.dictionary.Dictionary
        A dictionary object loaded following instructions in README.

    Returns
    -------
    float
        A float representing the prevalence (probability) of the specified topic in the given document.

    Notes
    -----
    If the topic ID is not present in the document's topic prevalence, the function returns 0.0.

    Examples
    --------
    >>> lda = gensim.models.ldamodel.LdaModel.load('model_path')
    >>> dictionary = gensim.corpora.dictionary.Dictionary.load('dictionary_path')
    >>> topic_score(2, 'Sample text of the document', lda, dictionary)
    0.15
    """

    prevalences = doc_topics(document, lda, dictionary)

    if tid not in prevalences.keys():
        return 0.0
    else:
        return prevalences[tid]


def ab_topic_diff_score(tid_a, tid_b, document, lda, dictionary):
    """
    Calculate a normalized score indicating the relative prevalence of two topics in a document.

    This function computes the difference between the topic scores for two topics,
    then normalizes by the sum of those scores, resulting in a metric that ranges
    from -1 to 1. A score of -1 indicates complete focus on `tid_b`, 1 indicates
    complete focus on `tid_a`, and 0 indicates no difference in prevalence between
    the two topics.

    Parameters
    ----------
    tid_a : int
        The ID of the first topic.
    tid_b : int
        The ID of the second topic.
    document : str
        A string containing a summary or article for which to calculate the score.
    lda : gensim.models.ldamodel.LdaModel
        A trained LDA model, loaded as per instructions in README.
    dictionary : gensim.corpora.dictionary.Dictionary
        A dictionary object loaded following instructions in README.

    Returns
    -------
    float
        A number between -1 and 1, where a higher value means higher prevalence of `tid_a` relative to `tid_b`, and 0 means equal prevalence.

    Notes
    -----
    If `tid_a` and `tid_b` are the same, the function returns 0. If both topic scores are 0, the function also returns 0.

    Examples
    --------
    >>> lda = gensim.models.ldamodel.LdaModel.load('model_path')
    >>> dictionary = gensim.corpora.dictionary.Dictionary.load('dictionary_path')
    >>> ab_topic_diff_score(1, 2, 'Sample text of the document', lda, dictionary)
    0.25
    """

    if tid_a == tid_b:
        return 0

    a = topic_score(tid_a, document, lda, dictionary)
    b = topic_score(tid_b, document, lda, dictionary)
    return 0 if (a == 0 and b == 0) else (a - b) / (a + b)
