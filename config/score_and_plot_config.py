"""
evaluation_config.py
This file contains the configuration for evaluation of the results. This includes the 
calculation of the scores and the configuration for the plots.
"""
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScoreAndPlotConfig:
    """
    Config class for storing settings and hyperparameters for evaluation of the results.
    Evaluation of the results comprises the calculation of the topic and quality scores for the 
    generated summaries ("results") and plotting these scores.
    """
    def __init__(self, **kwargs):
        """
        Initialize the ScoreAndPlotConfig class.
        """
        # Initialize the results scorer settings
        self._initialize_results_to_be_scored()
        self._initialize_scores_to_be_plotted()
        self._initialize_rouge_settings()
        self._initialize_topic_score_types()
        self._initialize_quality_score_types()

        # Initialize plot settings
        self._initialize_font_sizes_and_styles()
        self._initialize_method_dict()
        self._initialize_save_fig_settings()
        self._initialize_model_name_dict()
        self.fig_size = kwargs.get('fig_size', (8, 4))
        self.y_subtitle = kwargs.get('y_subtitle', 0.94)

    def _initialize_results_to_be_scored(self):
        """Initialize the results to be scored. """
        # choose from 'prompt_engineering', 'constant_shift', 'factor_scaling',
        # 'threshold_selection', 'topic_vectors'
        self.results_experiment_name = "constant_shift"
        # choose from available experiment results - without .json extension
        self.results_to_be_scored_file_name = "results_constant_shift_gemma_2b_3_80_90_1.json"

    def _initialize_scores_to_be_plotted(self):
        """Initialize the scores to be plotted. """
        # choose from 'prompt_engineering', 'constant_shift', 'factor_scaling',
        # 'threshold_selection', 'topic_vectors'
        self.scores_experiment_name = "constant_shift"
        # choose from available experiment scores - without .json extension
        self.scores_to_be_plotted_file_name = "constant_shift_gemma_2b_50_80_90_1.json"

    def _initialize_topic_score_types(self):
        """Initialize the topic score types. """
        self.all_topic_score_types = ['lemmatize', 'tokenize', 'dict', 'stem']
        self.topic_score_types = ['lemmatize', 'tokenize', 'dict']

    def _initialize_quality_score_types(self, **kwargs):
        """
        Initialize the quality score types.
        """
        self.quality_score_types: List[str] = ['mauve', 'bert', 'rougeL']
        self.all_quality_score_types: List[str] = self.quality_score_types.extend(self.rouge_methods)
        self.bert_model_name: str= kwargs.get('bert_model_name', 'microsoft/deberta-large-mnli')

    def _initialize_rouge_settings(self, **kwargs):
        self.rouge_methods = ['rouge1', 'rouge2', 'rougeL', 'avg_rouge']
        self.rouge_use_stemmer = True

    def _initialize_font_sizes_and_styles(self, **kwargs):
        """
        Initialize the font sizes for the plots.
        """
        self.fs_title: int = kwargs.get('fs_title', 14)
        self.style_title: str = kwargs.get('style_title', 'italic')
        self.fs_suptitle: int = kwargs.get('fs_suptitle', 16)
        self.fs_label: int = kwargs.get('fs_label', 12)
        self.fontweight_label: str = kwargs.get('fontweight_label', 'bold')
        self.fs_legend: int = kwargs.get('fs_legend', 12)
        self.fs_ticks: int = kwargs.get('fs_ticks', 10)

    def _initialize_method_dict(self, **kwargs):
        """
        Initialize the method dictionary for the plots.
        """
        self.method_dict = {
            'mauve': 'MAUVE',
            'bert': 'BERT',
            'avg_rouge': 'ROUGE',
            'rouge1': 'ROUGE-1',
            'rouge2': 'ROUGE-2',
            'rougeL': 'ROUGE-L',
            'stem' : 'Stemmed',
            'lemmatize' : 'Lemmatized',
            'dict' : 'Dictionary',
            'tokenize' : 'Tokenized',
        }

    def _initialize_save_fig_settings(self, **kwargs):
        """
        Initialize the save figure settings.
        """
        self.dpi = kwargs.get('dpi', 300)
        self.bbox_inches = kwargs.get('bbox_inches', 'tight')
        self.plot_data_format = kwargs.get('plot_data_format', 'pdf')

    def _initialize_model_name_dict(self, **kwargs):
        """
        Initialize the model name dictionary.
        """
        self.model_name_dict = {
            'gemma_2b' : '2B Gemma',
            'gemma 8B' : '8B Gemma',
            'llama_8b' : '8B Llama3',
            'llama_70b' : '70B Llama3',
        }
    #pylint: disable=line-too-long
    def validate(self):
        """Validate the configuration settings."""
        logger.info("Validating configuration settings...")
        assert isinstance(self.fig_size, tuple), "fig_size must be a tuple."
        assert isinstance(self.y_subtitle, float), "y_subtitle must be a float."
        assert isinstance(self.all_topic_score_types, list), "all_topic_score_types must be a list."
        assert isinstance(self.topic_score_types, list), "topic_score_types must be a list."
        assert isinstance(self.quality_score_types, list), "quality_score_types must be a list."
        assert isinstance(self.all_quality_score_types, list), "all_quality_score_types must be a list."
        assert isinstance(self.bert_model_name, str), "bert_model_name must be a string."
        assert isinstance(self.rouge_methods, list), "rouge_methods must be a list."
        assert isinstance(self.rouge_use_stemmer, bool), "rouge_use_stemmer must be a boolean."
        # ... TODO: continue 

        logging.info("Configuration settings are valid.")
    #pylint: enable=line-too-long