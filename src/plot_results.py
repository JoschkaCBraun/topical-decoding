'''
plot_results.py

This script plots topical and quality scores of experiment results. 
Plots are stored as .pdf files in the respective data/plots/{experiment_name} folder.
Results can also be plotted using the plot_results.ipynb notebook.
'''

# Standard library imports
import os
import sys
import json
import logging
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Third-party imports
import numpy as np
import torch

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

#pylint: disable=wrong-import-position
from config.score_and_plot_config import ScoreAndPlotConfig
from utils.load_and_get_utils import get_file_name, get_data_dir
#pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
class ResultsPlotter:
    """
    A class to plot the scores of the experiment results.
    """
    def __init__(self, eval_config: ScoreAndPlotConfig):
        """
        Initializes the ResultsPlotter class.
        """
        self.eval_config = eval_config
        self._initialize_paths()
        self._initialize_experiment_scores()
        self._initialize_experiment_settings()
        self._initialize_experiment_specific_values()
        self.eval_config = eval_config
        self._initialize_font_sizes_and_styles()
        self._initialize_fig_settings()
        self._initialize_dictionaries()
        self._initialize_file_name_and_directory()
        self._validate()

    def _initialize_paths(self):
        """Initialize the relevant paths. """
        try:
            self.data_folder_path = get_data_dir(os.getcwd())
            self.scores_folder_path = os.path.join(self.data_folder_path, 'scores')
            self.plots_folder_path = os.path.join(self.data_folder_path, 'plots')

        except FileNotFoundError as e:
            logger.error(f"Error: {e}. Please make sure that the data directory exists.")
            sys.exit(1)

    def _initialize_experiment_scores(self):
        experiment_scores_file_path = os.path.join(self.scores_folder_path,
                                                   self.eval_config.scores_experiment_name,
                                                   self.eval_config.scores_to_be_plotted_file_name)
        try:
            with open(experiment_scores_file_path, 'r', encoding='utf-8') as file:
                self.experiment_scores = json.load(file)
        except FileNotFoundError as e:
            logger.error(f"Error: {e}. Please make sure that the experiment scores file exists.")
            sys.exit(1)

    def _initialize_experiment_settings(self):
        """Initialize the experiment settings."""
        self.experiment_info = self.experiment_scores['experiment_info']
        self.experiment_config = self.experiment_info['EXPERIMENT_CONFIG']
        self.experiment_name = self.experiment_config['experiment_name']
        self.experiment_name_pretty = self.experiment_name.replace('_', ' ')
        self.model_alias = self.experiment_config['model_alias']
        self.num_articles = self.experiment_info['DATASET_CONFIG']['num_articles']
        self.num_beams = self.experiment_info['GENERATION_CONFIG']['num_beams']
        self.max_new_tokens = self.experiment_info['GENERATION_CONFIG']['max_new_tokens']
        self.min_new_tokens = self.experiment_info['GENERATION_CONFIG']['min_new_tokens']
        self.beam_info = "beam search" if self.num_beams > 1 else "greedy search"

    def _initialize_experiment_specific_values(self):
        """Initialize the experiment specific values."""
        if self.experiment_name == 'prompt_engineering':
            self.prompt_engineering_focus_types = self.experiment_info['PROMPT_ENGINEERING_DICT']['focus_types']
            self.focus_labels = [focus_type.replace('_', ' ') for focus_type in self.prompt_engineering_focus_types]
        elif self.experiment_name == 'constant_shift':
            self.factors = self.experiment_info['CONSTANT_SHIFT_DICT']['factors']
            self.factors_name = 'shift constants'
        elif self.experiment_name == 'factor_scaling':
            self.factors = self.experiment_info['FACTOR_SCALING_DICT']['factors']
            self.factors_name = 'scaling factors'
        elif self.experiment_name == 'threshold_selection':
            self.factors = self.experiment_info['THRESHOLD_SELECTION_DICT']['factors']
            self.topical_encouragement = self.experiment_info['THRESHOLD_SELECTION_DICT']['topical_encouragement']
            self.factors_name = 'selection thresholds'
        elif self.experiment_name == 'topic_vectors':
            self.topic_vectors_config = self.experiment_info['TOPIC_VECTORS_CONFIG']
            self.topic_encoding_types = list(self.topic_vectors_config.keys())
            self.factors_name = 'topic encoding types'
        else:
            raise ValueError(f"Invalid experiment name: {self.experiment_name}. "
                             "Please select a valid experiment name.")
    
    def _initialize_font_sizes_and_styles(self):
        """Initialize the font sizes for the plots. """
        self.fs_title = self.eval_config.fs_title
        self.style_title = self.eval_config.style_title
        self.fs_suptitle = self.eval_config.fs_suptitle
        self.fs_label = self.eval_config.fs_label
        self.fontweight_label = self.eval_config.fontweight_label
        self.fs_legend = self.eval_config.fs_legend
        self.fs_ticks = self.eval_config.fs_ticks
        self.y_subtitle = self.eval_config.y_subtitle
    
    def _initialize_fig_settings(self):
        """ Initialize the save figure settings. """
        self.dpi = self.eval_config.dpi
        self.bbox_inches = self.eval_config.bbox_inches
        self.plot_data_format = self.eval_config.plot_data_format
        self.fig_size = self.eval_config.fig_size

    def _initialize_dictionaries(self):
        """ Initialize the method dictionary for the plots. """
        self.method_dict = self.eval_config.method_dict
        self.model_name_dict = self.eval_config.model_name_dict
    
    def _initialize_file_name_and_directory(self): # check this function again!
        """ Initialize the file name. """
        file_info = {
            'output_type': 'plots',
            'experiment_name': self.experiment_name,
            'model_alias': self.model_alias,
            'num_articles': self.num_articles,
            'min_new_tokens': self.min_new_tokens,
            'max_new_tokens': self.max_new_tokens,
            'num_beams': self.num_beams,
        }
        self.file_name = get_file_name(file_info=file_info)

        self.experiment_plots_path = os.path.join(self.plots_folder_path, self.experiment_name)
    
    def _validate(self):
        """ Validate the experiment name. """
        if self.experiment_name not in self.VALID_EXPERIMENT_NAMES:
            logger.error(f"The run_experiments.py script is not configured to execute the experiment "
                         f'"{self.experiment_name}". It is only meant for the following experiments: '
                         f'{", ".join(self.VALID_EXPERIMENT_NAMES)}.')
            sys.exit(1)
        # self.eval_config.plot_experiment_name must be equal to self.experiment_name
        if self.eval_config.plot_experiment_name != self.experiment_name:
            logger.error(f"Invalid experiment name: {self.eval_config.plot_experiment_name}. "
                         "Please select a valid experiment name.")
            sys.exit(1)

    
    def plot_results(self, exhaustive: Optional[bool] = True, score_type: Optional[str] = None):
        """
        Plot the results of the experiment.
        """
        if exhaustive and score_type is None:
            if self.experiment_name == 'prompt_engineering':
                self.plot_prompt_engineering_results_exhaustive(
                    experiment_results=self.experiment_scores['experiment_results'],
                    score_type='topic')
                self.plot_prompt_engineering_results_exhaustive(
                    experiment_results=self.experiment_scores['experiment_results'],
                    score_type='quality')

            elif self.experiment_name in ['constant_shift', 'factor_scaling', 'threshold_selection']:
                self.plot_logits_reweighting_results_exhaustive(
                    experiment_results=self.experiment_scores['experiment_results'],
                    score_type='topic')
                self.plot_logits_reweighting_results_exhaustive(
                    experiment_results=self.experiment_scores['experiment_results'],
                    score_type='quality')
            
            elif self.experiment_name == 'topic_vectors':
                self.plot_topic_vectors_results_exhaustive(
                    experiment_results=self.experiment_scores['experiment_results'],
                    score_type='topic')
                self.plot_topic_vectors_results_exhaustive(
                    experiment_results=self.experiment_scores['experiment_results'],
                    score_type='quality')
            else:
                raise ValueError(f"Invalid experiment name: {self.experiment_name}. "
                                "Please select a valid experiment name.")
        else:
            logger.error("Currently not implemented yet. Therefore Invalid choice. "
                         "Please select 'exhaustive' as True and don't provide a 'score_type'.")

    def plot_prompt_engineering_results(self, avg_scores: dict, score_type: str, method: str):
        """
        Plot the average scores for different reweighting factors, selectable between topic scores and ROUGE scores.

        :param avg_scores: Dictionary of average scores with reweighting factors as keys.
        :param score_type: Type of scores to plot, either 'topic' or 'quality'.
        :param method: Method used to calculate the scores, e.g. 'MAUVE', 'BERTScore', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'.
        """
        score_type = score_type.lower()
        focus_types = self.prompt_engineering_focus_types

        if score_type == 'topic':
            wrt1 = 'tid1'
            wrt2 = 'tid2'
        elif score_type == 'quality':
            wrt1 = 'summary1'
            wrt2 = 'summary2'
        else:
            raise ValueError("Invalid choice. Please select 'topic' or 'quality'.")
        
        # set viridis colors
        colors = plt.cm.viridis(np.linspace(0, 1, 3))
        plt.rcParams["figure.figsize"] = self.fig_size
        avg_scores1 = [avg_scores[focus][wrt1] for focus in focus_types]
        avg_scores2 = [avg_scores[focus][wrt2] for focus in focus_types]

        x = np.arange(len(focus_types))
        width = 0.35
        
        plt.bar(x - width/2, avg_scores1, width, label=f'w.r.t. {wrt1}', color=colors[0], edgecolor='grey')
        plt.bar(x + width/2, avg_scores2, width, label=f'w.r.t. {wrt2}', color=colors[1], edgecolor='grey')
        plt.suptitle(f'Summary {score_type} scores of prompt engineering experiment',
                     fontsize=self.fs_suptitle, fontweight='bold', y=self.y_subtitle)
        plt.title(f'Using {self.model_name_dict[self.model_alias]} model with {self.beam_info} on {self.num_articles} articles',
                  fontsize=self.fs_title, style=self.style_title)
        if score_type == 'topic':
            plt.ylabel(f'Average {score_type} score ({self.method_dict[method].lower()})', fontsize=self.fs_label, fontweight=self.fontweight_label)
        elif score_type == 'quality':
            plt.ylabel(f'Average {self.method_dict[method]} score', fontsize=self.fs_label, fontweight=self.fontweight_label)
        plt.xlabel('Summary type', fontsize=self.fs_label, fontweight=self.fontweight_label)
        plt.xticks(x, self.focus_labels, fontsize=self.fs_ticks)
        plt.legend(prop={'size': self.fs_legend})
        plt.tight_layout()
        plt.savefig(fname=f'{self.experiment_plots_path}/{score_type.lower()}_{method}_scores_{self.file_name}.pdf',
                    dpi=self.dpi, format=self.plot_data_format)
        plt.show()

    def plot_prompt_engineering_results_exhaustive(self, experiment_results: dict, score_type: str):
        """
        Plot the average scores for different reweighting factors, selectable between topic scores and ROUGE scores.
        :param experiment_results: Dictionary containing the scores for each method and focus type.
        :param score_type: String indicating whether to plot 'topic' or 'quality'.
        """
        score_type = score_type.lower()

        # Initialize data storage
        scores_data = {}
        focus_types = ['tid1_focus', 'no_focus', 'tid2_focus']
        methods = []

        if score_type == 'topic':
            wrt1, wrt2 = 'tid1', 'tid2'
            methods = ['lemmatize', 'tokenize', 'dict']
        elif score_type == 'quality':
            wrt1, wrt2 = 'summary1', 'summary2'
            methods = ['mauve', 'bert', 'rougeL']
        else:
            raise ValueError("Invalid choice for 'score_type', please select 'topic' or 'quality'.")
        
        for method in methods:
            scores_data[method] = calculate_scores(experiment_results=experiment_results, score_type=score_type, method=method)

        plt.rcParams["figure.figsize"] = self.fig_size
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        index = np.arange(len(focus_types))
        bar_width = 0.25
        bottom_heights = np.zeros(len(focus_types)*2)
        for m, method in enumerate(methods):
            for i, focus in enumerate(focus_types):
                score_1 = scores_data[method][focus][wrt1]
                score_2 = scores_data[method][focus][wrt2]
                plt.bar(x=i - bar_width/1.5, height=score_1, width=bar_width,
                        bottom=bottom_heights[2*i+0], label=f'{method}' if i == 0 else "",
                        color=colors[m], edgecolor='black', linewidth=2)
                plt.bar(x=i + bar_width/1.5, height=score_2, width=bar_width,
                        bottom=bottom_heights[2*i+1], color=colors[m],
                        edgecolor='black', linestyle='--', linewidth=2)
                bottom_heights[2*i+0] += score_1
                bottom_heights[2*i+1] += score_2

        plt.suptitle(f'Summary {score_type} scores of prompt engineering experiment', fontsize=self.fs_suptitle, fontweight='bold', y=self.y_subtitle)
        plt.title(f'Using {self.model_name_dict[self.model_alias]} model with {self.beam_info} on {self.num_articles} articles',
                  fontsize=self.fs_title, style=self.style_title)
        plt.ylabel(f'{score_type.capitalize()} Scores', fontsize=self.fs_label, fontweight=self.fontweight_label)
        plt.ylim(top=max(bottom_heights) * 1.2)
        plt.xlabel('Summary Type', fontsize=self.fs_label, fontweight=self.fontweight_label)
        plt.xticks(index, self.focus_labels, fontsize=self.fs_ticks)
        
        legend_elements = []
        legend_elements.append(Patch(facecolor='white', edgecolor='black', label=f"w.r.t {wrt1}"))
        legend_elements.append(Patch(facecolor='white', edgecolor='black', linestyle='--', label=f"w.r.t. {wrt2}"))
        legend_elements.append(Patch(facecolor='white', alpha=0, label=""))
        for m, method in enumerate(methods):
            legend_elements.append(Patch(facecolor=colors[m], label=f"{self.method_dict[method]}"))

        plt.legend(handles=legend_elements, prop={'size': self.fs_legend}, loc='upper right', ncol=2, frameon=False)
        plt.tight_layout()
        plt.savefig(fname=f'{self.experiment_plots_path}/{score_type.lower()}_exhaustive_{self.file_name}.pdf',
                    dpi=self.dpi, format=self.plot_data_format)
        plt.show()

    def plot_logits_reweighting_results(self, avg_scores: dict, score_type: str, method: str):
        """
        Plot the average scores for different reweighting factors, selectable between topic scores and ROUGE scores.

        :param avg_scores: Dictionary of average scores with reweighting factors as keys.
        :param choice: String indicating whether to plot 'topic' or 'rouge'
        """
        labels = [f'{factor}' for factor in avg_scores.keys()]
        scores = [score for score in avg_scores.values()]
        colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))

        # Plotting
        plt.figure(figsize=self.fig_size)
        plt.bar(x=labels, height=scores, color=colors, edgecolor='grey')
        plt.xlabel(xlabel=self.factors_name.capitalize(), fontweight=self.fontweight_label, fontsize=self.fs_label)
        plt.suptitle(f'Summary {score_type} scores of {self.experiment_name_pretty} experiment', fontsize=self.fs_suptitle, fontweight='bold', y=self.y_subtitle)
        plt.title(f'Using {self.model_name_dict[self.model_alias]} model with {self.beam_info} on {self.num_articles} articles',
                  fontsize=self.fs_title, style=self.style_title)
        if score_type == 'topic':
            plt.ylabel(f'Average {score_type} score', fontsize=self.fs_label, fontweight=self.fontweight_label)
        elif score_type == 'quality':
            plt.ylabel(f'Average {self.method_dict[method]} score', fontsize=self.fs_label, fontweight=self.fontweight_label)
        plt.tight_layout()
        plt.savefig(fname=f'{self.experiment_plots_path}/{score_type.lower()}_{method}_scores_{self.file_name}.pdf',
                    dpi=self.dpi, format=self.plot_data_format)
        plt.show()
        
    def plot_logits_reweighting_results_exhaustive(self, experiment_results: dict, score_type: str):
        """
        Plot the average scores for different reweighting factors, selectable between topic scores and ROUGE scores.
        :param experiment_results: Dictionary containing the scores for each method and focus type.
        :param score_type: String indicating whether to plot 'topic' or 'quality'.
        """
        score_type = score_type.lower()
        scores_data = {}
        if score_type == 'topic':
            methods = ['lemmatize', 'tokenize', 'dict']
        elif score_type == 'quality':
            methods = ['mauve', 'bert', 'rougeL']
        else:
            raise ValueError("Invalid choice for 'score_type', please select 'topic' or 'quality'.")
        
        for method in methods:
            scores_data[method] = calculate_scores(experiment_results=experiment_results, score_type=score_type, method=method)
        factors = [factor for factor in scores_data[methods[0]].keys()]
        labels = [f'{factor}' for factor in factors]

        plt.rcParams["figure.figsize"] = self.fig_size
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        index = np.arange(len(factors))
        bar_width = 0.7
        bottom_heights = np.zeros(len(factors))
        for m, method in enumerate(methods):
            for i, factor in enumerate(factors):
                score = scores_data[method][factor]
                plt.bar(x=i, height=score, width=bar_width, bottom=bottom_heights[i],
                        label=f"{self.method_dict[method]} Score" if i == 0 else "", color=colors[m],
                        edgecolor='black', linewidth=1)
                bottom_heights[i] += score
        
        plt.ylabel(f'{score_type.capitalize()} Scores', fontsize=self.fs_label, fontweight=self.fontweight_label)
        plt.suptitle(f'Summary {score_type} scores of {self.experiment_name_pretty} experiment', fontsize=self.fs_suptitle, fontweight='bold', y=self.y_subtitle)
        plt.title(f'Using {self.model_name_dict[self.model_alias]} model with {self.beam_info} on {self.num_articles} articles',
                  fontsize=self.fs_title, style=self.style_title)
        
        plt.ylim(top=max(bottom_heights) * 1.2)
        plt.xlabel(xlabel=self.factors_name.capitalize(), fontweight=self.fontweight_label, fontsize=self.fs_label)
        plt.xticks(ticks=index, labels=factors, fontsize=self.fs_ticks)
        plt.legend(prop={'size': self.fs_legend}, loc='upper center', ncol=3, frameon=False)
        plt.tight_layout()
        plt.savefig(fname=f'{self.experiment_plots_path}/{score_type.lower()}_exhaustive_{self.file_name}.pdf',
                    dpi=self.dpi, format=self.plot_data_format)
        plt.show()

    def plot_topic_vectors_results(self, avg_scores: dict, score_type: str, method: str):
        """
        Plot the average scores for different reweighting factors, selectable between topic scores and ROUGE scores.

        :param avg_scores: Dictionary of average scores with reweighting factors as keys.
        :param choice: String indicating whether to plot 'topic' or 'rouge'
        """
        labels = [topic_encoding_type for topic_encoding_type in avg_scores.keys()]
        scores = [score for score in avg_scores.values()]
        colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))

        # Plotting
        plt.figure(figsize=self.fig_size)
        plt.bar(x=labels, height=scores, color=colors, edgecolor='grey')
        plt.xlabel(xlabel=self.factors_name.capitalize(), fontweight=self.fontweight_label, fontsize=self.fs_label)
        plt.xticks(fontsize=self.fs_ticks, rotation=5)
        plt.suptitle(f'Summary {score_type} scores of {self.experiment_name_pretty} experiment', fontsize=self.fs_suptitle, fontweight='bold', y=self.y_subtitle)
        plt.title(f'Using {self.model_name_dict[self.model_alias]} model with {self.beam_info} on {self.num_articles} articles',
                  fontsize=self.fs_title, style=self.style_title)
        if score_type == 'topic':
            plt.ylabel(f'Average {score_type} score', fontsize=self.fs_label, fontweight=self.fontweight_label)
        elif score_type == 'quality':
            plt.ylabel(f'Average {self.method_dict[method]} score', fontsize=self.fs_label, fontweight=self.fontweight_label)
        plt.tight_layout()
        plt.savefig(fname=f'{self.experiment_plots_path}/{score_type.lower()}_{method}_scores_{self.file_name}.pdf',
                    dpi=self.dpi, format=self.plot_data_format)
        plt.show()

    def plot_topic_vectors_results_exhaustive(self, experiment_results: dict, score_type: str):
        """
        Plot the average scores for different reweighting factors, selectable between topic scores and ROUGE scores.
        :param experiment_results: Dictionary containing the scores for each method and focus type.
        :param score_type: String indicating whether to plot 'topic' or 'quality'.
        """
        score_type = score_type.lower()
        scores_data = {}
        if score_type == 'topic':
            methods = ['lemmatize', 'tokenize', 'dict']
        elif score_type == 'quality':
            methods = ['mauve', 'bert', 'rougeL']
        else:
            raise ValueError("Invalid choice for 'score_type', please select 'topic' or 'quality'.")
        
        for method in methods:
            scores_data[method] = calculate_scores(experiment_results=experiment_results, score_type=score_type, method=method)
        factors = [factor for factor in scores_data[methods[0]].keys()]
        labels = [topic_encoding_type for topic_encoding_type in avg_scores.keys()]

        plt.rcParams["figure.figsize"] = self.fig_size
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        index = np.arange(len(factors))
        bar_width = 0.7
        bottom_heights = np.zeros(len(factors))
        for m, method in enumerate(methods):
            for i, factor in enumerate(factors):
                score = scores_data[method][factor]
                plt.bar(x=i, height=score, width=bar_width, bottom=bottom_heights[i],
                        label=f"{self.method_dict[method]} Score" if i == 0 else "", color=colors[m],
                        edgecolor='black', linewidth=1)
                bottom_heights[i] += score
        
        plt.ylabel(f'{score_type.capitalize()} Scores', fontsize=self.fs_label, fontweight=self.fontweight_label)
        plt.suptitle(f'Summary {score_type} scores of {self.experiment_name_pretty} experiment', fontsize=self.fs_suptitle, fontweight='bold', y=self.y_subtitle)
        plt.title(f'Using {self.model_name_dict[self.model_alias]} model with {self.beam_info} on {self.num_articles} articles',
                  fontsize=self.fs_title, style=self.style_title)
        
        plt.ylim(top=max(bottom_heights) * 1.2)
        plt.xlabel(xlabel=self.factors_name.capitalize(), fontweight=self.fontweight_label, fontsize=self.fs_label)
        plt.xticks(ticks=index, labels=factors, fontsize=self.fs_ticks, rotation=5)
        plt.legend(prop={'size': self.fs_legend}, loc='upper center', ncol=3, frameon=False)
        plt.tight_layout()
        plt.savefig(fname=f'{self.experiment_plots_path}/{score_type.lower()}_exhaustive_{self.file_name}.pdf',
                    dpi=self.dpi, format=self.plot_data_format)
        plt.show()

def main():
    '''
    Generate plots for the results of the experiments.
    '''
    eval_config = ScoreAndPlotConfig()
    results_plotter = ResultsPlotter(eval_config=eval_config)

    if results_plotter.experiment_name not in results_plotter.VALID_EXPERIMENT_NAMES:
        logger.error('The run_experiments.py script is not configured to execute the experiment '
                     f'"{eval_config.EXPERIMENT_NAME}". It is only meant for the following experiments: '
                     f'{", ".join(eval_config.VALID_EXPERIMENT_NAMES)}.')
        sys.exit(1)

    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Generate abstractive summaries, settings are defined in the config.py file

    results_plotter.plot_results()

# pylint: enable=logging-fstring-interpolation

if __name__ == "__main__":
    main()