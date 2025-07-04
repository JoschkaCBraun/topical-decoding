'''
evaluation_utils.py
This script contains functions for evaluating the topical relevance of generated summaries.
'''

# Standard library imports
import os
import sys
import string
import logging
from typing import Dict, List, Any

# Third-party imports
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as nltk_stopwords
import spacy
import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, BertTokenizer, BertModel
from bert_score import BERTScorer
from evaluate import load
import torch

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from config import DATASET_CONFIG, TOPICS_CONFIG, ROUGE_METRICS
from utils.read_and_load_utils import load_lda, load_dictionary, read_dataset, load_tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""Initializes necessary resources."""
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
lemmatizer = spacy.load('en_core_web_sm', disable=["parser", "ner"])
stemmer = SnowballStemmer(language='english')
stopwords = set(nltk_stopwords.words('english'))
punctuation = set(string.punctuation)

def get_topic_words(lda: LdaModel, tid: int, num_topic_words: int) -> List[str]:
    """
    Returns the top words for a given topic from the LDA model.
    
    :param lda: The trained LDA model.
    :param tid: The topic number to get the top words for.
    :param num_topic_words: The number of top words to return.
    :return: A list of top words for the specified topic.
    """
    try:
        # Note: num_words here limits the number of words returned for the topic.
        topic_words = lda.show_topic(topicid=tid, topn=num_topic_words)
        topic_words = [word for word, prob in topic_words]
        return topic_words
    except Exception as e:
        print(f"Error in getting top topic words: {e}")
        return []

def get_word_variations(word: str) -> List[str]:
    """
    Generates variations of a word, including different capitalizations and spaces.
    
    :param word: The word for which to generate variations.
    :return: A list of word variations.
    """
    lemmatized_word = lemmatizer(word)[0].lemma_ if len(lemmatizer(word)) > 0 else word
    stemmed_word = stemmer.stem(word)
    variations = {word, lemmatized_word, stemmed_word}
    variations_capitalization = set()
    for word in variations:
        variations_capitalization.add(word.lower())
        variations_capitalization.add(word.capitalize())
    variations_spaces = set()
    for word in variations_capitalization:
        variations_spaces.add(word)
        variations_spaces.add(' ' + word)
        variations_spaces.add(word + ' ')
        variations_spaces.add(' ' + word + ' ')
    return list(variations_spaces)

def get_topic_tokens(tokenizer: AutoTokenizer, lda: LdaModel, tid: int, num_topic_words: int,
                     ) -> List[int]:
    """
    Retrieves token IDs for the top words associated with a given topic from an LDA model.

    :param tokenizer: The tokenizer used to convert words to token IDs.
    :param lda: The trained LDA model.
    :param topic_id: The ID of the topic for which to retrieve top words.
    :param num_topic_words: The number of top words to retrieve for the specified topic.
    :return: A list of unique token IDs corresponding to the top words of the specified topic.
    """
    topic_words = get_topic_words(lda=lda, tid=tid, num_topic_words=num_topic_words)
    special_token_ids = set(tokenizer.all_special_ids)
    token_ids_set = set()

    for word in topic_words:
        word_variations = get_word_variations(word)
        word_variations = [word for word in word_variations if word not in stopwords]
        token_ids = tokenizer(word_variations)['input_ids']
        token_ids = set([token for sublist in token_ids for token in sublist])
        token_ids = [token_id for token_id in token_ids if token_id not in special_token_ids]
        token_ids_set.update(token_ids)
    return list(token_ids_set)

def calculate_scores(experiment_results: dict, score_type: str, method: str) -> dict:
    """
    Calculates average scores for the specified experiment results.
    
    :param experiment_results: Dictionary containing experiment information and a list of summaries.
    :param score_type: The type of score to calculate ('topic' or 'quality').
    :param method: Method used for topic or quality score calculation.
    :return: Dictionary of average scores for the specified experiment.
    """
    score_type = score_type.lower()
    if score_type not in ['topic', 'quality']:
        raise ValueError("Invalid score_type. Choose either 'topic' or 'quality'.")
    method = method.lower()
    method = 'rougeL' if method == 'rougel' else method
    if score_type == 'topic' and method not in ['dict', 'tokenize', 'lemmatize', 'stem']:
        raise ValueError(f"Invalid method '{method}' for score_type 'topic'. Choose either 'dict', 'tokenize', 'lemmatize', or 'stem'.")
    if score_type == 'quality' and method not in ['avg_rouge', 'rouge1', 'rouge2', 'rougeL', 'mauve', 'bert']:
        raise ValueError(f"Invalid method '{method}' for score_type 'quality'. Choose either 'avg_rouge', 'rouge1', 'rouge2', 'rougeL', 'mauve', or 'bert'.")
    experiment_name = experiment_results['experiment_information']['EXPERIMENT_CONFIG']['experiment_name']

    if experiment_name == 'baseline':
        if score_type == 'topic':
            scores = baseline_topic_scores(experiment_results=experiment_results, method=method)
        elif score_type == 'quality':
            scores = baseline_quality_scores(experiment_results=experiment_results, method=method)

    elif experiment_name in ['factor_scaling', 'constant_shift', 'threshold_selection']:
        if experiment_name == 'constant_shift':
            factors = experiment_results['experiment_information']['CONSTANT_SHIFT_DICT']['shift_constants']
        if experiment_name == 'factor_scaling':
            factors = experiment_results['experiment_information']['FACTOR_SCALING_DICT']['scaling_factors']
        if experiment_name == 'threshold_selection':
            factors = experiment_results['experiment_information']['THRESHOLD_SELECTION_DICT']['selection_thresholds']
        if score_type == 'topic':
            scores = logits_reweighting_topic_scores(experiment_results=experiment_results,
                                                     method=method, factors=factors)
        if score_type == 'quality':
            scores = logits_reweighting_quality_scores(experiment_results=experiment_results,
                                                       method=method, factors=factors)
    else:
        raise ValueError("Invalid experiment name. Choose between 'baseline', 'factor_scaling',"/
                         "'constant_shift', or 'threshold_selection'.")
    return scores

def baseline_topic_scores(experiment_results: dict, method: str) -> Dict[str, float]:
    """
    Calculate average topic scores for the three types of summaries for both tid1 and tid2.
    
    :param experiment_results: Dictionary containing experiment info and list of summaries,
                               each containing summaries for an article and its topic IDs.
    :param method: Method used for text processing before calculating topic scores.
    :return: Dictionary of scores categorized by focus type and topic ID.
    """
    scores = dict()
    lda = load_lda()
    focus_types = experiment_results['experiment_information']['BASELINE_DICT']['focus_types']

    for focus in focus_types:
        summaries = [summary[focus] for summary in experiment_results['generated_summaries']]
        tids_tid1 = [summary['tid1'] for summary in experiment_results['generated_summaries']]
        tids_tid2 = [summary['tid2'] for summary in experiment_results['generated_summaries']]

        scores_tid1 = calculate_topic_scores(experiment_results=experiment_results, lda=lda,
                                             texts=summaries, tids=tids_tid1, method=method)
        scores_tid2 = calculate_topic_scores(experiment_results=experiment_results, lda=lda,
                                             texts=summaries, tids=tids_tid2, method=method)
        score_tid1 = np.mean(scores_tid1)
        score_tid2 = np.mean(scores_tid2)
        scores[focus] = {"tid1": score_tid1, "tid2": score_tid2}
    return scores

def baseline_quality_scores(experiment_results: dict, method: str) -> Dict[str, Dict[str, float]]:
    """
    Calculates the average quality scores across all summaries for each focus type and each reference summary.

    :param experiment_results: Dictionary containing experiment information and a list of summaries, 
                      each containing different types of summaries for each article.
    :param method: Method for the quality score calculation.
    """
    dataset_name = experiment_results['experiment_information']['DATASET_CONFIG']['dataset_name']
    reference_summaries = read_dataset(dataset_name)
    focus_types = experiment_results['experiment_information']['BASELINE_DICT']['focus_types']
    quality_scores = dict()

    if method == 'mauve':
        mauve = load(path='mauve')
    elif method == 'bert':
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        bert = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', lang='en',
                          rescale_with_baseline=True, device=device)
    elif method in ['avg_rouge', 'rouge1', 'rouge2', 'rougeL']:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    else:
        raise ValueError("Invalid method. Choose 'avg_rouge', 'rouge1', 'rouge2', 'rougeL', 'mauve', or 'bert'.")

    for focus in focus_types:
        generated_summaries = [summary[focus] for summary in experiment_results['generated_summaries']]
        reference_summaries1 = [reference_summaries.iloc[idx]['summary1'] for idx in range(len(generated_summaries))]
        reference_summaries2 = [reference_summaries.iloc[idx]['summary2'] for idx in range(len(generated_summaries))]

        # Compute ROUGE scores for all summaries against both reference summaries
        if method in ['avg_rouge', 'rouge1', 'rouge2', 'rougeL']:
            quality_scores_1 = calculate_rouge_scores(text_predictions=generated_summaries,
                                                      text_targets=reference_summaries1,
                                                      method=method, scorer=scorer)
            quality_scores_2 = calculate_rouge_scores(text_predictions=generated_summaries,
                                                      text_targets=reference_summaries2,
                                                      method=method, scorer=scorer)
        elif method == 'mauve':
            quality_scores_1 = calculate_mauve_scores(text_predictions=generated_summaries,
                                                      text_targets=reference_summaries1,
                                                      mauve=mauve)
            quality_scores_2 = calculate_mauve_scores(text_predictions=generated_summaries,
                                                      text_targets=reference_summaries2,
                                                      mauve=mauve)
        elif method == 'bert':
            quality_scores_1 = calculate_bert_scores(text_predictions=generated_summaries,
                                                     text_targets=reference_summaries1, bert=bert)
            quality_scores_2 = calculate_bert_scores(text_predictions=generated_summaries,
                                                     text_targets=reference_summaries2, bert=bert)

        quality_score_1 = np.mean(quality_scores_1)
        quality_score_2 = np.mean(quality_scores_2)
        quality_scores[focus] = {"summary1": quality_score_1, "summary2": quality_score_2}

    return quality_scores

def logits_reweighting_topic_scores(experiment_results: dict, method: str, factors: list) -> Dict[str, float]:
    """
    Calculate average topic score for all summaries generated with the same reweighting factor.
    
    :param experiment_results: Dictionary containing experiment info and list of summaries, 
                               each containing summaries for an article and its topic IDs.
    :param method: Method used for text processing before calculating topic scores.
    :return: Dictionary of scores categorized by focus type and topic ID.
    """
    scores = dict()
    lda = load_lda()
    for factor in factors:
        summaries = [summary[str(factor)] for summary in experiment_results['generated_summaries']]
        tids = [summary['tid1'] for summary in experiment_results['generated_summaries']]
        scores_factor = calculate_topic_scores(experiment_results=experiment_results, lda=lda,
                                               texts=summaries, tids=tids, method=method)
        score_factor = np.mean(scores_factor)
        scores[factor] = score_factor
    return scores

def logits_reweighting_quality_scores(experiment_results: dict, factors: list, method: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores for different reweighting factors across summaries.

    :param experiment_results: Dictionary containing experimenr results.
    :param factors: List of reweighting factors.
    :return: Dictionary of ROUGE scores categorized by reweighting factor.
    """
    quality_scores = dict()
    dataset_name = experiment_results['experiment_information']['DATASET_CONFIG']['dataset_name']
    reference_texts = read_dataset(dataset_name)

    if method == 'mauve':
        mauve = load(path='mauve')
    elif method == 'bert':
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        bert = BERTScorer(model_type='microsoft/deberta-large-mnli', lang='en',
                          rescale_with_baseline=False, device=device)
    elif method in ['avg_rouge', 'rouge1', 'rouge2', 'rougeL']:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    else:
        raise ValueError("Invalid method. Choose 'avg_rouge', 'rouge1', 'rouge2', 'rougeL', 'mauve', or 'bert'.")
    
    for factor in factors:
        generated_summaries = [summary[str(factor)] for summary in experiment_results['generated_summaries']]
        reference_summaries = [reference_texts.iloc[idx]['summary1'] for idx in range(len(generated_summaries))]

        if method in ['avg_rouge', 'rouge1', 'rouge2', 'rougeL']:
            factor_scores= calculate_rouge_scores(text_predictions=generated_summaries,
                                                  text_targets=reference_summaries,
                                                  method=method, scorer=scorer)
        elif method == 'mauve':
            factor_scores = calculate_mauve_scores(text_predictions=generated_summaries,
                                                   text_targets=reference_summaries, mauve=mauve)
        elif method == 'bert':
            factor_scores = calculate_bert_scores(text_predictions=generated_summaries,
                                                  text_targets=reference_summaries, bert=bert)
        factor_score = np.mean(factor_scores)
        quality_scores[factor] = factor_score
    return quality_scores

def calculate_topic_scores(experiment_results: dict, lda: LdaModel, texts: List[str],
                           tids: List[int], method: str) -> List[float]:
    """
    Calculates topic scores for a list of documents.

    :param experiment_results: Dictionary containing experiment information.
    :param texts: A list of texts (strings).
    :param tids: A list of topic IDs.
    :param method: Method for text processing ('dict', 'lemmatize', or 'stem').
    :return: A list of topic scores.
    """
    if len(texts) != len(tids):
        raise ValueError("The length of documents and tids must be the same.")

    logger.info("Calculating topic scores with method: %s", method)
    num_topic_words = experiment_results['experiment_information']['TOPICS_CONFIG']['num_topic_words']

    if method == 'dict':
        return calculate_dict_topic_scores(texts=texts, lda=lda, tids=tids)

    elif method == 'tokenize':
        model_alias = experiment_results['experiment_information']['EXPERIMENT_CONFIG']['model_alias']
        return calculate_tokenizer_topic_scores(texts=texts, lda=lda, tids=tids,
                                                num_topic_words=num_topic_words,
                                                model_alias=model_alias)

    elif method in ['lemmatize', 'stem']:
        return calculate_stem_or_lem_topic_scores(texts=texts, lda=lda, tids=tids, method=method,
                                                  num_topic_words=num_topic_words)
    else:
        raise ValueError("Invalid method. Choose 'dict', 'tokenize', 'lemmatize', or 'stem'.")

def calculate_rouge_scores(text_predictions: List[str], text_targets: List[str], method: str,
                           scorer: rouge_scorer.RougeScorer) -> List[float]:
    """
    Calculate ROUGE scores for batches of text_predictions (generated summaries) with respect to 
    text_targets (reference summaries).
    
    :param generated_summaries: List of generated summary texts.
    :param reference_summaries: List of reference summary texts.
    :param method: The method for calculating ROUGE scores ('rouge1', 'rouge2', 'rougeL', or 'avg_rouge').
    :return: List of scores corresponding to the specified ROUGE method or the average ROUGE score.
    """
    logger.info("Calculating ROUGE scores...")
    rouge_scores = []
    for predicted_text, target_text in zip(text_predictions, text_targets):
        scores = scorer.score(target=target_text, prediction=predicted_text)
        if method == 'avg_rouge':
            average_score = np.mean([value.fmeasure for value in scores.values()])
            rouge_scores.append(average_score)
        else:
            rouge_scores.append(scores[method].fmeasure)
    return rouge_scores

def calculate_mauve_scores(text_predictions: List[str], text_targets: List[str], mauve: Any) -> List[float]:
    """
    Calculate MAUVE scores for batches of text_predictions (generated summaries) with respect to 
    text_targets (reference summaries).
    
    :param generated_summaries: List of generated summary texts.
    :param reference_summaries: List of reference summary texts.
    :param mauve: The MAUVE object.
    :return: List of MAUVE scores.
    """
    logger.info("Calculating MAUVE scores...")
    mauve_outputs = mauve.compute(predictions=text_predictions, references=text_targets)
    if mauve_outputs is None:
        raise ValueError("Error calculating MAUVE score.")
    return mauve_outputs.mauve

def calculate_bert_scores(text_predictions: List[str], text_targets: List[str], bert: BERTScorer) -> List[float]:
    """
    Calculate BERT scores for batches of text_predictions (generated summaries) with respect to 
    text_targets (reference summaries).
    
    :param generated_summaries: List of generated summary texts.
    :param reference_summaries: List of reference summary texts.
    :return: List of BERT scores.
    """
    logger.info("Calculating BERT scores...")
    _, _, f1 = bert.score(cands=text_predictions, refs=text_targets)
    return f1.tolist()

def calculate_dict_topic_scores(texts: List[str], tids: List[int], lda: LdaModel) -> List[float]:
    """
    Calculates topic scores using a dictionary method for a list of documents, 
    each against a corresponding topic ID.

    :param texts: A list of documents.
    :param tids: A list of topic IDs, one for each document.
    :return: A list of topic scores, one for each document.
    """
    dictionary = load_dictionary()
    vec_bows = [dictionary.doc2bow(text.lower().split()) for text in texts]
    document_topics = [dict(lda.get_document_topics(vec_bow)) for vec_bow in vec_bows]
    return [prevalences.get(tid, 0.0) for prevalences, tid in zip(document_topics, tids)]

def calculate_tokenizer_topic_scores(texts: List[str], lda: LdaModel, tids: List[int],
                                     num_topic_words: int, model_alias: str) -> List[float]:
    """
    Calculate topic scores for a list of documents using a tokenizer.

    :param texts: A list of documents.
    :param lda: A trained LDA model.
    :param tids: A list of topic IDs.
    :param num_topic_words: The number of top words for each topic.
    :param model_alias: The alias of the model used to generate summaries.
    :return: A list of topic scores.
    """
    tokenizer = load_tokenizer(model_alias=model_alias)
    scores = []
    for text, tid in zip(texts, tids):
        topic_tokens = set(get_topic_tokens(tokenizer=tokenizer, lda=lda, tid=tid,
                                            num_topic_words=num_topic_words))
        words = text.lower().split()
        tokenized_text = []
        for word in words:
            word_tokens = tokenizer(word, return_tensors='pt')['input_ids'][0].tolist()
            tokenized_text.extend(word_tokens)
        score = float(sum([1 for token in tokenized_text if token in topic_tokens]))
        normalized_score = score / float(len(tokenized_text))
        scores.append(normalized_score)
    return scores

def calculate_stem_or_lem_topic_scores(texts: List[str], lda: LdaModel, tids: List[int],
                                       num_topic_words: int, method: str) -> List[float]:
    """
    Calculate topic scores for a list of documents using either stemming or lemmatization,
    excluding stopwords and punctuation.

    :param texts: A list of documents.
    :param lda: A trained LDA model.
    :param tids: A list of topic IDs.
    :param num_topic_words: The number of top words for each topic.
    :param method: The method for text processing ('lemmatize' or 'stem').
    :return: A list of topic scores.
    """

    def process_text(text: str, method: str):
        text.lower()
        processed_words = set()
        if method == 'lemmatize':
            lemmatized_text = lemmatizer(text)
            processed_words = {token.lemma_ for token in lemmatized_text if not 
                               (token.is_punct or token.is_space or token.text in stopwords)}
        elif method == 'stem':
            words = text.split()
            processed_words = {stemmer.stem(word) for word in words if word not in stopwords}
        return processed_words

    processed_texts = [process_text(text, method) for text in texts]
    scores = []

    for doc, tid in zip(processed_texts, tids):
        topic_words_with_weights = dict(lda.show_topic(tid, topn=num_topic_words))
        total_weight = sum(topic_words_with_weights.values())
        if total_weight == 0:
            scores.append(0.0)
            continue
        weighted_score = sum([topic_words_with_weights.get(word, 0) for word in doc])
        normalized_score = weighted_score / total_weight
        scores.append(normalized_score)
    return scores

def doc_topics(document: str, lda: LdaModel, dictionary: Dictionary) -> Dict[int, float]:
    """
    Calculate the prevalence of various topics within a given document using an LDA model.

    :param document: A summary or article for which to calculate topics.
    :param lda: A trained LDA model.
    :param dictionary: A dictionary object.
    :return: A dictionary with keys as topic IDs and values as the prevalence of that topic in the
    document.
    """
    lda.minimum_phi_value = TOPICS_CONFIG['min_phi_value']
    lda.per_word_topics = False
    vec_bow = dictionary.doc2bow(document.split(" "))
    temp = lda[vec_bow]
    temp.sort(key=lambda x: x[1], reverse=True)
    return dict(temp)

def ab_topic_diff_score(tid_a: int, tid_b: int, document: str, lda: LdaModel,
                        dictionary: Dictionary) -> float:
    """
    Calculate a normalized score indicating the relative prevalence of two topics in a document.

    This function computes the difference between the topic scores for two topics, then normalizes 
    by the sum of those scores, resulting in a metric that ranges from -1 to 1. A score of -1 
    indicates complete focus on `tid_b`, 1 indicates complete focus on `tid_a`, and 0 indicates 
    no difference in prevalence between the two topics.

    :param tid_a : The topic id of the first topic of the document.
    :param tid_b : The topic id of the second topic of the document.
    :param document : A string containing a summary or article for which to calculate the score.
    :param lda : A trained LDA model
    :param dictionary : A dictionary object loaded following instructions in README.
    :return: A float between -1 and 1, where a higher value means higher prevalence of `tid_a` 
    relative to `tid_b`, and 0 means equal prevalence.
    """

    if tid_a == tid_b:
        return 0.0

    prevalences = doc_topics(document, lda, dictionary)
    a = prevalences.get(tid_a, 0.0)
    b = prevalences.get(tid_b, 0.0)

    return 0 if (a == 0.0 and b == 0.0) else (a - b) / (a + b)

def main():
    '''Main function.'''

if __name__ == "__main__":
    main()
