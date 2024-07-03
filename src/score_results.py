'''
score_results.py

This script evaluates given results of an experiment. The generated abstractive summaries of
articles are analysed for their topic focus and linguistic quality.
Calculated Scores are stored as json files in the respective data/scores/{experiment_name} folder.
'''

# Standard library imports
import os
import sys
import json
import logging
from typing import Optional, List, Dict, Union, Any

# Third-party imports
import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from evaluate import load
import torch

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

#pylint: disable=wrong-import-position
from src.utils.load_and_get_utils import load_dictionary, load_dataset, load_tokenizer,\
    get_topic_tokens, get_device, load_lda, store_results_or_scores, get_data_dir
from src.utils.text_processing_utils import TextProcessor
from config.score_and_plot_config import ScoreAndPlotConfig
from config.experiment_config import ExperimentConfig
#pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
class ResultsScorer:
    """
    A class to score the results of an experiment.
    """
    def __init__(self, eval_config: ScoreAndPlotConfig):
        """
        Initializes the ResultsEvaluator class.
        """
        self.eval_config = eval_config
        self._initialize_paths()
        self._initialize_experiment_results()
        self.experiment_information = self.experiment_results['experiment_info']
        self._initialize_experiment_config()
        self._initialize_generation_config()
        self._initialize_dataset_config()
        self._initialize_topics_config()
        
        self.generated_summaries = self.experiment_results['generated_summaries']
        
        self.experiment_config = ExperimentConfig()
        self._initialize_experiment_names()
        self._initialize_score_types()
        self._initialize_evaluation_scorer()
        self.tokenizer = load_tokenizer(self.eval_config)
        self.device = get_device()
        self.lda : LdaModel = load_lda()
        self.dictionary : Dictionary = load_dictionary()
        self.text_processor = TextProcessor()

    def _initialize_paths(self):
        """Initialize the relevant paths. """
        try:
            self.data_folder_path = get_data_dir(os.getcwd())
            self.results_folder_path = os.path.join(self.data_folder_path, 'results')
            self.scores_folder_path = os.path.join(self.data_folder_path, 'scores')
        
        except FileNotFoundError as e:
            logger.error(f"Error: {e}. Please make sure that the data directory exists.")
            sys.exit(1)
    
    def _initialize_experiment_results(self):
        experiment_results_file_path = os.path.join(self.results_folder_path,
                                                    self.eval_config.results_experiment_name,
                                                    self.eval_config.results_to_be_scored_file_name)
        try:
            with open(experiment_results_file_path, 'r', encoding='utf-8') as file:
                self.experiment_results = json.load(file)
        except FileNotFoundError as e:
            logger.error(f"Error: {e}. Please make sure that the experiment scores file exists.")
            sys.exit(1)

    def _initialize_experiment_config(self):
        """ Initialize experiment related configurations. """
        # TODO: Have to resolve this possible name conflict with ExperimentConfig()
        self.experiment_name = self.experiment_information['EXPERIMENT_CONFIG']['experiment_name']
        self.model_alias = self.experiment_information['EXPERIMENT_CONFIG']['model_alias']

    def _initialize_generation_config(self):
        """ Initialize generation related configurations. """
        self.generation_config = self.experiment_information['GENERATION_CONFIG']
        self.min_new_tokens = self.generation_config['min_new_tokens']
        self.max_new_tokens = self.generation_config['max_new_tokens']
        self.num_beams = self.generation_config['num_beams']
        self.do_sample = self.generation_config['do_sample']
        self.top_p = self.generation_config['top_p']
        self.top_k = self.generation_config['top_k']

    def _initialize_dataset_config(self):
        """ Initialize dataset related configurations. """
        self.dataset_config = self.experiment_information['DATASET_CONFIG']
        self.training_dataset_name = self.dataset_config['training_dataset_name']
        self.test_dataset_name = self.dataset_config['test_dataset_name']
        self.num_articles = self.dataset_config['num_articles']
        self.batch_size = self.dataset_config['batch_size']
        self.num_workers = self.dataset_config['num_workers']
        self.shuffle = self.dataset_config['shuffle']

    def _initialize_topics_config(self):
        """ Initialize topics related configurations. """
        self.topics_config = self.experiment_information['TOPICS_CONFIG']
        self.num_topic_words = self.topics_config['num_topic_words']
        self.min_phi_value = self.topics_config['min_phi_value']

    def _initialize_experiment_names(self):
        """ Initialize the experiment names. """
        self.valid_experiment_names = self.experiment_config['VALID_EXPERIMENT_NAMES']
        self.logits_reweighting_experiment_names = self.experiment_config['LOGITS_REWEIGHTING_EXPERIMENT_NAMES']
        self.prompt_engineering_focus_types = self.experiment_config['PROMPT_ENGINEERING_CONFIG']['focus_types']

    def _initialize_score_types(self):
        """ Initialize the score types. """
        self.topic_score_types = self.eval_config.topic_score_types
        self.quality_score_types = self.eval_config.quality_score_types
        self.all_topic_score_types = self.eval_config.all_topic_score_types
        self.all_quality_score_types = self.eval_config.all_quality_score_types
        self.rouge_methods = self.eval_config.rouge_methods

    def _initialize_evaluation_scorer(self):
        """ Initialize the evaluation scorer. """
        self.mauve = load(path='mauve')
        self.bert = BERTScorer(model_type='microsoft/deberta-large-mnli', lang='en',
                               rescale_with_baseline=False, device=self.device)
        self.rouge= rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                             use_stemmer=self.eval_config.rouge_use_stemmer)
        
    def score_results(self):
        """
        Evaluate the results of an experiment.
        """
        logger.info("Started scoring results...")
        # first check if plot is already generated
        # if not generate plot. For this check if scores are already calculated
        # if not calculate scores and then generate plot. For this check if summaries are already generated
        # if not generate summaries and then calculate scores and then generate plot.
        logger.info("Calculating topic scores...")
        topic_scores = dict()
        for method in self.all_topic_score_types:
            scores = self.calculate_scores(score_type='topic', method=method)
            topic_scores[method] = scores
    
        logger.info("Calculating quality scores...")
        quality_scores = dict()
        for method in self.all_quality_score_types:
            scores = self.calculate_scores(score_type='quality', method=method)
            quality_scores[method] = scores
        
        scores = {'topic_scores': topic_scores, 'quality_scores': quality_scores}
        self.store_scores(scores=scores)
        
    def calculate_scores(self, score_type: str, method: str) -> dict:
        """
        Calculates average scores for the specified experiment results.
        
        :param score_type: The type of score to calculate ('topic' or 'quality').
        :param method: Method used for topic or quality score calculation.
        :return: Dictionary of average scores for the specified experiment.
        """
        score_type = score_type.lower()
        if score_type not in ['topic', 'quality']:
            raise ValueError("Invalid score_type. Choose either 'topic' or 'quality'.")
        method = method.lower()
        method = 'rougeL' if method == 'rougel' else method
        if score_type == 'topic' and method not in self.all_topic_score_types:
            raise ValueError(f"Invalid method '{method}' for score_type 'topic'. Choose either"\
                             f" {self.enumerate_options(self.all_topic_score_types)}.")
        if score_type == 'quality' and method not in self.all_quality_score_types:
            raise ValueError(f"Invalid method '{method}' for score_type 'quality'. Choose either"\
                             f"{self.enumerate_options(self.all_quality_score_types)}.")
        if self.experiment_name == 'prompt_engineering':
            if score_type == 'topic':
                scores = self.prompt_engineering_topic_scores(method=method)
            elif score_type == 'quality':
                scores = self.prompt_engineering_quality_scores(method=method)

        elif self.experiment_name in self.experiment_config.LOGITS_REWEIGHTING_EXPERIMENT_NAMES:
            if self.experiment_name == 'constant_shift':
                factors = self.experiment_information['CONSTANT_SHIFT_DICT']['shift_constants']
            if self.experiment_name == 'factor_scaling':
                factors = self.experiment_information['FACTOR_SCALING_DICT']['scaling_factors']
            if self.experiment_name == 'threshold_selection':
                factors = self.experiment_information['THRESHOLD_SELECTION_DICT']['selection_thresholds']
            if score_type == 'topic':
                scores = self.logits_reweighting_topic_scores(method=method, factors=factors)
            if score_type == 'quality':
                scores = self.logits_reweighting_quality_scores(method=method, factors=factors)
        elif self.experiment_name == 'topic_vectors':
            topic_encoding_types = self.experiment_information['TOPIC_VECTORS_CONFIG'].keys()
            if score_type == 'topic':
                scores = self.logits_reweighting_topic_scores(method=method,
                                                              factors=topic_encoding_types)
            elif score_type == 'quality':
                scores = self.logits_reweighting_quality_scores(method=method,
                                                                factors=topic_encoding_types)
        else:
            raise ValueError("Invalid experiment name. Choose either 'prompt_engineering', "\
                             "'factor_scaling', 'constant_shift', 'threshold_selection' or "\
                             "topic_vectors."\
                             "No other experiments are implemented by calculate_scores.")
        return scores

    def prompt_engineering_topic_scores(self, method: str) -> Dict[str, float]:
        """
        Calculate average topic scores for the three types of summaries for both tid1 and tid2.

        :param method: Method used for text processing before calculating topic scores.
        :return: Dictionary of scores categorized by focus type and topic ID.
        """
        scores = dict()
        focus_types = self.experiment_information['PROMPT_ENGINEERING_DICT']['focus_types']

        for focus in focus_types:
            summaries = [summary[focus] for summary in self.generated_summaries]
            tids_tid1 = [summary['tid1'] for summary in self.generated_summaries]
            tids_tid2 = [summary['tid2'] for summary in self.generated_summaries]

            scores_tid1 = self.calculate_topic_scores(texts=summaries, tids=tids_tid1, method=method)
            scores_tid2 = self.calculate_topic_scores(texts=summaries, tids=tids_tid2, method=method)
            score_tid1 = np.mean(scores_tid1)
            score_tid2 = np.mean(scores_tid2)
            scores[focus] = {"tid1": score_tid1, "tid2": score_tid2}
        return scores

    def prompt_engineering_quality_scores(self, method: str) -> Dict[str, Dict[str, float]]:
        """
        Calculates the average quality scores across all summaries for each focus type and each reference summary.

        :param experiment_results: Dictionary containing experiment information and a list of summaries, 
                        each containing different types of summaries for each article.
        :param method: Method for the quality score calculation.
        """
        reference_summaries = load_dataset(self.test_dataset_name)
        focus_types = self.experiment_information['PROMPT_ENGINEERING']['focus_types']
        quality_scores = dict()

        if method not in self.all_quality_score_types:
            raise ValueError("Invalid method. Choose 'avg_rouge', 'rouge1', 'rouge2', 'rougeL', 'mauve', or 'bert'.")
        
        for focus in focus_types:
            generated_summaries = [summary[focus] for summary in generated_summaries]
            reference_summaries1 = [reference_summaries.iloc[idx]['summary1'] for idx in range(len(generated_summaries))]
            reference_summaries2 = [reference_summaries.iloc[idx]['summary2'] for idx in range(len(generated_summaries))]

            # Compute ROUGE scores for all summaries against both reference summaries
            if method in ['avg_rouge', 'rouge1', 'rouge2', 'rougeL']:
                quality_scores_1 = self.calculate_rouge_scores(text_predictions=generated_summaries,
                                                               text_targets=reference_summaries1,
                                                               method=method)
                quality_scores_2 = self.calculate_rouge_scores(text_predictions=generated_summaries,
                                                               text_targets=reference_summaries2,
                                                               method=method)
            elif method == 'mauve':
                quality_scores_1 = self.calculate_mauve_scores(text_predictions=generated_summaries,
                                                               text_targets=reference_summaries1)
                quality_scores_2 = self.calculate_mauve_scores(text_predictions=generated_summaries,
                                                               text_targets=reference_summaries2)
            elif method == 'bert':
                quality_scores_1 = self.calculate_bert_scores(text_predictions=generated_summaries,
                                                              text_targets=reference_summaries1)
                quality_scores_2 = self.calculate_bert_scores(text_predictions=generated_summaries,
                                                              text_targets=reference_summaries2)

            quality_score_1 = np.mean(quality_scores_1)
            quality_score_2 = np.mean(quality_scores_2)
            quality_scores[focus] = {"summary1": quality_score_1, "summary2": quality_score_2}

        return quality_scores

    def logits_reweighting_topic_scores(self, method: str, factors: list) -> Dict[str, float]:
        """
        Calculate average topic score for all summaries generated with the same reweighting factor.
        
        :param method: Method used for text processing before calculating topic scores.
        :return: Dictionary of scores categorized by focus type and topic ID.
        """
        scores = dict()
        for factor in factors:
            summaries = [summary[str(factor)] for summary in self.generated_summaries]
            tids = [summary['tid1'] for summary in self.generated_summaries]
            scores_factor = self.calculate_topic_scores(texts=summaries, tids=tids, method=method)
            score_factor = np.mean(scores_factor)
            scores[factor] = score_factor
        return scores

    def logits_reweighting_quality_scores(self, factors: list, method: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores for different reweighting factors across summaries.

        :param factors: List of reweighting factors.
        :return: Dictionary of ROUGE scores categorized by reweighting factor.
        """
        quality_scores = dict()
        reference_texts = load_dataset(self.test_dataset_name)

        if method not in self.all_quality_score_types:
            raise ValueError(f"Invalid method. Choose from: {self.enumerate_options(self.all_quality_score_types)}.")
        
        for factor in factors:
            generated_summaries = [summary[str(factor)] for summary in self.generated_summaries]
            reference_summaries = [reference_texts.iloc[idx]['summary1'] for idx in range(len(generated_summaries))]

            if method in self.rouge_methods:
                factor_scores= self.calculate_rouge_scores(text_predictions=generated_summaries,
                                                           text_targets=reference_summaries,
                                                           method=method)
            elif method == 'mauve':
                factor_scores = self.calculate_mauve_scores(text_predictions=generated_summaries,
                                                            text_targets=reference_summaries)
            elif method == 'bert':
                factor_scores = self.calculate_bert_scores(text_predictions=generated_summaries,
                                                           text_targets=reference_summaries)
            factor_score = np.mean(factor_scores)
            quality_scores[factor] = factor_score
        return quality_scores

    def calculate_topic_scores(self, texts: List[str], tids: List[int], method: str) -> List[float]:
        """
        Calculates topic scores for a list of documents.

        :param texts: A list of texts (strings).
        :param tids: A list of topic IDs.
        :param method: Method for text processing ('dict', 'lemmatize', or 'stem').
        :return: A list of topic scores.
        """
        if len(texts) != len(tids):
            raise ValueError("The length of documents and tids must be the same.")

        logger.info("Calculating topic scores with method: %s", method)

        if method == 'dict':
            return self.calculate_dict_topic_scores(texts=texts, tids=tids)

        elif method == 'tokenize':
            return self.calculate_tokenizer_topic_scores(texts=texts, tids=tids)

        elif method in ['lemmatize', 'stem']:
            return self.calculate_stem_or_lem_topic_scores(texts=texts, tids=tids, method=method)
        else:
            raise ValueError("Invalid method. Choose 'dict', 'tokenize', 'lemmatize', or 'stem'.")

    def calculate_rouge_scores(self, text_predictions: List[str], text_targets: List[str],
                               method: str) -> List[float]:
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
            scores = self.rouge.score(target=target_text, prediction=predicted_text)
            if method == 'avg_rouge':
                average_score = np.mean([value.fmeasure for value in scores.values()])
                rouge_scores.append(average_score)
            else:
                rouge_scores.append(scores[method].fmeasure)
        return rouge_scores

    def calculate_mauve_scores(self, text_predictions: List[str], text_targets: List[str]) -> List[float]:
        """
        Calculate MAUVE scores for batches of text_predictions (generated summaries) with respect to 
        text_targets (reference summaries).
        
        :param generated_summaries: List of generated summary texts.
        :param reference_summaries: List of reference summary texts.
        :param mauve: The MAUVE object.
        :return: List of MAUVE scores.
        """
        logger.info("Calculating MAUVE scores...")
        mauve_outputs = self.mauve.compute(predictions=text_predictions, references=text_targets)
        if mauve_outputs is None:
            raise ValueError("Error calculating MAUVE score.")
        return mauve_outputs.mauve

    def calculate_bert_scores(self, text_predictions: List[str], text_targets: List[str]) -> List[float]:
        """
        Calculate BERT scores for batches of text_predictions (generated summaries) with respect to 
        text_targets (reference summaries).
        
        :param generated_summaries: List of generated summary texts.
        :param reference_summaries: List of reference summary texts.
        :return: List of BERT scores.
        """
        logger.info("Calculating BERT scores...")
        _, _, f1 = self.bert.score(cands=text_predictions, refs=text_targets)
        return f1.tolist()

    def calculate_dict_topic_scores(self, texts: List[str], tids: List[int]) -> List[float]:
        """
        Calculates topic scores using a dictionary method for a list of documents, 
        each against a corresponding topic ID.

        :param texts: A list of documents.
        :param tids: A list of topic IDs, one for each document.
        :return: A list of topic scores, one for each document.
        """
        vec_bows = [self.dictionary.doc2bow(text.lower().split()) for text in texts]
        document_topics = [dict(self.lda.get_document_topics(vec_bow)) for vec_bow in vec_bows]
        return [prevalences.get(tid, 0.0) for prevalences, tid in zip(document_topics, tids)]

    def calculate_tokenizer_topic_scores(self, texts: List[str], tids: List[int]) -> List[float]:
        """
        Calculate topic scores for a list of documents using a tokenizer.

        :param texts: A list of documents.
        :param tids: A list of topic IDs.
        :return: A list of topic scores.
        """
        scores = []
        for text, tid in zip(texts, tids):
            topic_tokens = set(get_topic_tokens(tokenizer=self.tokenizer, lda=self.lda, tid=tid,
                                                num_topic_words=self.num_topic_words))
            words = text.lower().split()
            tokenized_text = []
            for word in words:
                word_tokens = self.tokenizer(word, return_tensors='pt')['input_ids'][0].tolist()
                tokenized_text.extend(word_tokens)
            score = float(sum([1 for token in tokenized_text if token in topic_tokens]))
            normalized_score = score / float(len(tokenized_text))
            scores.append(normalized_score)
        return scores

    def calculate_stem_or_lem_topic_scores(self, texts: List[str], tids: List[int], method: str
                                           ) -> List[float]:
        """
        Calculate topic scores for a list of documents using either stemming or lemmatization,
        excluding stopwords and punctuation.

        :param texts: A list of documents.
        :param lda: A trained LDA model.
        :param tids: A list of topic IDs.
        :param method: The method for text processing ('lemmatize' or 'stem').
        :return: A list of topic scores.
        """

        processed_texts = [self.text_processor.process_text(text, method) for text in texts]
        scores = []

        for doc, tid in zip(processed_texts, tids):
            topic_words_with_weights = dict(self.lda.show_topic(tid, topn=self.num_topic_words))
            total_weight = sum(topic_words_with_weights.values())
            if total_weight == 0:
                scores.append(0.0)
                continue
            weighted_score = sum([topic_words_with_weights.get(word, 0) for word in doc])
            normalized_score = weighted_score / total_weight
            scores.append(normalized_score)
        return scores

    def doc_topics(self, document: str) -> Dict[int, float]:
        """
        Calculate the prevalence of various topics within a given document using an LDA model.

        :param document: A summary or article for which to calculate topics.
        :return: A dictionary with keys as topic IDs and values as the prevalence of that topic in the
        document.
        """
        self.lda.minimum_phi_value = self.min_phi_value
        self.lda.per_word_topics = False
        vec_bow = self.dictionary.doc2bow(document.split(" "))
        temp = self.lda[vec_bow]
        temp.sort(key=lambda x: x[1], reverse=True)
        return dict(temp)

    def ab_topic_diff_score(self, tid_a: int, tid_b: int, document: str) -> float:
        """
        Calculate a normalized score indicating the relative prevalence of two topics in a document.

        This function computes the difference between the topic scores for two topics, then normalizes 
        by the sum of those scores, resulting in a metric that ranges from -1 to 1. A score of -1 
        indicates complete focus on `tid_b`, 1 indicates complete focus on `tid_a`, and 0 indicates 
        no difference in prevalence between the two topics.

        :param tid_a : The topic id of the first topic of the document.
        :param tid_b : The topic id of the second topic of the document.
        :param document : A string containing a summary or article for which to calculate the score.
        :return: A float between -1 and 1, where a higher value means higher prevalence of `tid_a` 
        relative to `tid_b`, and 0 means equal prevalence.
        """

        if tid_a == tid_b:
            return 0.0

        prevalences = self.doc_topics(document=document)
        a = prevalences.get(tid_a, 0.0)
        b = prevalences.get(tid_b, 0.0)

        return 0 if (a == 0.0 and b == 0.0) else (a - b) / (a + b)

    def enumerate_options(self, options: List[str]) -> str:
        """
        Enumerate the options in a list.

        :param options: A list of strings.
        :return: A string with the enumerated options.
        """
        if len(options) == 1:
            return f'({options[0]})'
        elif len(options) == 2:
            return f'({options[0]} or {options[1]})'
        else:
            return f'({", ".join(options[:-1])}, or {options[-1]})'

    def store_scores(self, scores: Dict[str, dict]) -> None:
        """
        Stores the generated summaries to a file in the results directory.

        :param scores: A dictionary containing the scores.
        """
        self.experiment_results['scores'] = scores

        file_info = {
            'output_type': 'scores',
            'experiment_name': self.experiment_name,
            'model_alias': self.model_alias,
            'num_articles': self.num_articles,
            'min_new_tokens': self.min_new_tokens,
            'max_new_tokens': self.max_new_tokens,
            'num_beams': self.num_beams,
        }
        store_results_or_scores(file=self.experiment_results, file_info=file_info)
        return

def main():
    '''
    Evaluate the results of an experiment.
    '''
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Generate abstractive summaries, settings are defined in the config.py file
    results_scorer = ResultsScorer(eval_config=ScoreAndPlotConfig())
    results_scorer.score_results()

# pylint: enable=logging-fstring-interpolation

if __name__ == "__main__":
    main()