'''
run_experiments.py

This script contains the SummaryGenerator class that generates abstractive summaries of articles
from the NEWTS dataset. The experiment type and hyperparameters can be chosen in the config.py file.

Result are stored in JSON files in the respective data/results_{experiment_name} folder.
Results can be plotted using the plot_results.ipynb notebook.
'''

# Standard library imports
import os
import sys
import logging
from typing import Optional, List, Dict, Union
# Third-party imports
import torch

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

#pylint: disable=wrong-import-position
from config.experiment_config import ExperimentConfig
from experiments.logits_reweighting.logits_reweighting import CustomModel
from experiments.topic_vectors.create_topic_vectors import get_topic_vector
from utils.load_and_get_utils import get_topic_words, load_model_and_tokenizer, \
    get_dataloader, load_lda, get_topic_tokens, store_results_or_scores, get_data_dir
#pylint: enable=wrong-import-position

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pylint: disable=logging-fstring-interpolation
class SummaryGenerator:
    """
    A class to generate abstractive summaries of articles from the NEWTS dataset.
    """
    def __init__(self, config: ExperimentConfig):
        """
        Initializes the SummaryGenerator class.
        """
        self.config = config
        self._initialize_experiment_config()
        self._initialize_generation_config()
        self._initialize_dataset_config()
        self._initialize_topics_config()
        self.experiment_info = self.get_experiment_info()

        self.valid_experiment_names = self.config.VALID_EXPERIMENT_NAMES
        self.logits_reweighting_experiment_names = self.config.LOGITS_REWEIGHTING_EXPERIMENT_NAMES
        self.data_dir = get_data_dir(os.getcwd())
        self.tokenizer, self.model, self.device = self._load_model_and_tokenizer()
        self.max_length = self.get_max_position_embeddings()
        self.lda = load_lda()
        self.dataloader = get_dataloader(dataset_name=config.TEST_DATASET_NAME, config=self.config)
        self.generation_config_object = self.config.get_generation_config(model_alias=self.model_alias,
                                                                          tokenizer=self.tokenizer)

    def _initialize_experiment_config(self):
        self.experiment_config = self.config.EXPERIMENT_CONFIG
        self.experiment_name = self.experiment_config['experiment_name']
        self.model_alias = self.experiment_config['model_alias']

    def _initialize_generation_config(self):
        """ Initialize generation related configurations. """
        self.generation_config = self.config.GENERATION_CONFIG
        self.min_new_tokens = self.generation_config['min_new_tokens']
        self.max_new_tokens = self.generation_config['max_new_tokens']
        self.num_beams = self.generation_config['num_beams']
        self.do_sample = self.generation_config['do_sample']
        self.top_p = self.generation_config['top_p']
        self.top_k = self.generation_config['top_k']

    def _initialize_dataset_config(self):
        """ Initialize dataset related configurations. """
        self.dataset_config = self.config.DATASET_CONFIG
        self.training_dataset_name = self.dataset_config['training_dataset_name']
        self.test_dataset_name = self.dataset_config['test_dataset_name']
        self.num_articles = self.dataset_config['num_articles']
        self.batch_size = self.dataset_config['batch_size']
        self.num_workers = self.dataset_config['num_workers']
        self.shuffle = self.dataset_config['shuffle']

    def _initialize_topics_config(self):
        """ Initialize topics related configurations. """
        self.topics_config = self.config.TOPICS_CONFIG
        self.num_topic_words = self.topics_config['num_topic_words']
        self.min_phi_value = self.topics_config['min_phi_value']

    def _load_model_and_tokenizer(self):
        """
        Load the model and tokenizer for generating summaries.
        """
        if self.experiment_name in self.logits_reweighting_experiment_names:
            tokenizer, model, device = load_model_and_tokenizer(config=self.config,
                                                                CustomModel=CustomModel)
            model.model.eval()
        else:
            tokenizer, model, device = load_model_and_tokenizer(config=self.config)
            model.eval()
        return tokenizer, model, device
 
    def get_experiment_info(self):
        """
        Get information about the experiment configuration and set experiment-specific parameters.
        """
        config = self.config
        experiment_info = {
            'EXPERIMENT_CONFIG': self.experiment_config,
            'GENERATION_CONFIG': self.generation_config,
            'DATASET_CONFIG': self.dataset_config,
            'TOPICS_CONFIG': self.topics_config,
        }
        self.topical_encouragement = None

        if self.experiment_name == 'baseline':
            experiment_info['BASELINE_CONFIG'] = config.BASELINE_CONFIG
        elif self.experiment_name == 'prompt_engineering':
            experiment_info['PROMPT_ENGINEERING_CONFIG'] = config.PROMPT_ENGINEERING_CONFIG
            self.focus_types = config.PROMPT_ENGINEERING_CONFIG['focus_types']
        elif self.experiment_name == 'factor_scaling':
            experiment_info['FACTOR_SCALING_CONFIG'] = config.FACTOR_SCALING_CONFIG
            self.factors = config.FACTOR_SCALING_CONFIG['scaling_factors']
        elif self.experiment_name == 'constant_shift':
            experiment_info['CONSTANT_SHIFT_CONFIG'] = config.CONSTANT_SHIFT_CONFIG
            self.factors = config.CONSTANT_SHIFT_CONFIG['shift_constants']
        elif self.experiment_name == 'threshold_selection':
            experiment_info['THRESHOLD_SELECTION_CONFIG'] = config.THRESHOLD_SELECTION_CONFIG
            self.factors = config.THRESHOLD_SELECTION_CONFIG['selection_thresholds']
            self.topical_encouragement = config.THRESHOLD_SELECTION_CONFIG['topical_encouragement']
        elif self.experiment_name == 'topic_vectors':
            experiment_info['TOPIC_VECTORS_CONFIG'] = config.TOPIC_VECTORS_CONFIG
        else:
            logger.error("Invalid experiment_name: %s set in config.py.", self.experiment_name)
            raise ValueError(f"Invalid experiment_name: {self.experiment_name} set in config.py.")
        return experiment_info
    
    def generate_summaries(self):
        """
        Generates topical abstractive summaries for a set of articles.
        """
        try:
            logger.info(f"Started generating summaries using {self.model_alias} model for "\
                        f"{self.experiment_name} experiment.")
            
            generated_summaries = []
            with torch.no_grad():  # Operations inside don't track gradients
                for i, batch in enumerate(self.dataloader):
                    if i * self.batch_size >= self.num_articles:
                        break

                    for index in range(len(batch['article'])):
                        generated_summaries.append(self.generate_individual_summary(batch=batch,
                                                                                    index=index))

            logger.info(f"Generated {len(generated_summaries)} summaries in total"\
                        f" for {self.num_articles} articles.")
            self.store_results(summaries=generated_summaries)
            return

        except Exception as e:
            logger.error(f"Error generating summaries with {self.model_alias} model for "\
                         f"{self.experiment_name} experiment: {e}")
            raise

    def generate_individual_summary(self, batch: Dict, index: int) -> Dict[str, Union[int, str]]:
        '''
        Generate summaries for a single article.

        :param batch: The batch of articles.
        :param index: The index of the article in the batch.
        :return: A dictionary containing the generated summaries.
        '''
        article, article_idx = batch['article'][index], batch['article_idx'][index].item()
        tid1, tid2 = batch['tid1'][index].item(), batch['tid2'][index].item()
        article_summaries = {'article_idx': article_idx, 'tid1': tid1}

        prompt = self.generate_prompt(article=article, tid=None)
        tokenized_prompt = self.tokenize_prompt(prompt=prompt)

        if self.experiment_name == 'baseline':
            outputs = self.generate_summary(tokenized_prompt=tokenized_prompt)
            decoded_summary = self.decode_summary(output=outputs, tokenized_prompt=tokenized_prompt)
            article_summaries['baseline'] = decoded_summary

        elif self.experiment_name == 'prompt_engineering':
            article_summaries['tid2'] = tid2

            for focus_type in self.focus_types:
                tid = tid1 if focus_type == 'tid1_focus' else (tid2 if focus_type == 'tid2_focus'
                                                               else None)
                prompt = self.generate_prompt(article=article, tid=tid)
                tokenized_prompt = self.tokenize_prompt(prompt=prompt)
                outputs = self.generate_summary(tokenized_prompt=tokenized_prompt)
                decoded_summary = self.decode_summary(output=outputs,
                                                      tokenized_prompt=tokenized_prompt)
                article_summaries[focus_type] = decoded_summary

        elif self.experiment_name in self.logits_reweighting_experiment_names:
            topic_tokens = get_topic_tokens(tokenizer=self.tokenizer, lda=self.lda, tid=tid1,
                                            config=self.config)

            for factor in self.factors:
                output = self.model.generate(
                    experiment_name=self.experiment_name,
                    input_ids=tokenized_prompt['input_ids'],
                    attention_mask=tokenized_prompt['attention_mask'],
                    topic_token_ids=topic_tokens,
                    scaling_factor=factor,
                    shift_constant=factor,
                    selection_threshold=factor,
                    topical_encouragement=self.topical_encouragement,
                    **self.generation_config_object.to_dict())

                # summary = tokenizer.decode(output[tokenized_prompt['input_ids'].shape[1]:],
                # skip_special_tokens=False)
                # decoded_summary = self.tokenizer.decode(output.squeeze(),
                # skip_special_tokens=False)
                decoded_summary = self.decode_summary(output=output,
                                                      tokenized_prompt=tokenized_prompt)
                article_summaries[str(factor)] = decoded_summary

        elif self.experiment_name == 'topic_vectors':
            for topic_encoding_type in self.config.TOPIC_VECTORS_CONFIG:
                steering_vector = get_topic_vector(config=self.config, tid=tid1,
                                                   topic_encoding_type=topic_encoding_type,
                                                    data_dir=self.data_dir)
                with steering_vector.apply(self.model):
                    outputs = self.generate_summary(tokenized_prompt=tokenized_prompt)
                decoded_summary = self.decode_summary(output=outputs,
                                                      tokenized_prompt=tokenized_prompt)
                article_summaries[topic_encoding_type] = decoded_summary

        return article_summaries

    def get_max_position_embeddings(self) -> int:
        """
        Get the maximum length for tokenization based on model alias.
        """
        if self.model_alias in ['openelm_270m', 'openelm_450m', 'openelm_1b', 'openelm_3b']:
            return 2048
        elif self.experiment_name in self.logits_reweighting_experiment_names:
            return self.model.model.config.max_position_embeddings
        else:
            return self.model.config.max_position_embeddings

    def tokenize_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        Tokenizes the prompt for model input.
        
        :param prompt: The prompt to be tokenized.
        :return: A dictionary containing the tokenized prompt.
        """
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True,
                                          truncation=True,
                                          max_length=self.max_length).to(self.device)
        return tokenized_prompt

    def generate_summary(self, tokenized_prompt: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generates the summary using the model.
        
        :param tokenized_prompt: The tokenized prompt for the model input.
        :return: The model output tensor.
        """
        output = self.model.generate(input_ids=tokenized_prompt['input_ids'],
                                     attention_mask=tokenized_prompt['attention_mask'],
                                     **self.generation_config_object.to_dict())
        return output

    def decode_summary(self, output: torch.Tensor, tokenized_prompt: Dict[str, torch.Tensor]
                       ) -> str:
        """
        Decodes the model output to generate the summary.
        
        :param output: The model output tensor.
        :param tokenized_prompt: The tokenized prompt for the model input.
        :return: A string containing the decoded summary.
        """
        decoded_summary = self.tokenizer.decode(
            output[:, tokenized_prompt['input_ids'].shape[1]:].squeeze(),
            skip_special_tokens=False)
        return decoded_summary

    def generate_prompt(self, article: str, tid: Optional[int] = None) -> str:
        """
        Constructs a prompt for model, optionally with a focus on specific topic words.
        
        :param article: The article text to be summarized.
        :param tid: The topic identifier for the focus topic, if any.
        :return: A string containing the structured prompt for summary generation.
        """

        try:
            initial_instruction = "Generate a summary of the text" # "Summarize the text"
            if tid is not None:
                top_words = get_topic_words(lda=self.lda, tid=tid, config=self.config)
                topic_description = ", ".join(top_words)
                topical_instruction = f" focussing on {topic_description}."
            else:
                topical_instruction = "."
            prompt = f'{initial_instruction}{topical_instruction}\n"{article}"\n'
            # "\nAs stated previously, {initial_instruction.lower()}{topical_instruction}'
            if self.model_alias == 'mistral_7b':
                prompt = f"<s>[INST] {prompt} [/INST]"
            return prompt
        except Exception as e:
            logger.error(f"Error generating prompt_engineering prompts: {e}")
            raise

    def store_results(self, summaries: List[Dict[str, str]]) -> None:
        """
        Stores the generated summaries to a file in the results directory.

        :param summaries: The generated summaries to store.
        """
        results = {'experiment_info': self.experiment_info,
                   'generated_summaries': summaries}
        
        file_info = {
            'output_type': 'results',
            'experiment_name': self.experiment_name,
            'model_alias': self.model_alias,
            'num_articles': self.num_articles,
            'min_new_tokens': self.min_new_tokens,
            'max_new_tokens': self.max_new_tokens,
            'num_beams': self.num_beams,
        }

        store_results_or_scores(file=results, file_info=file_info)
        return

def main():
    '''
    Generate abstractive summaries of articles from the NEWTS dataset.
    '''
    config = ExperimentConfig()

    if config.EXPERIMENT_NAME not in config.VALID_EXPERIMENT_NAMES:
        logger.error('The run_experiments.py script is not configured to execute the experiment '
                     f'"{config.EXPERIMENT_NAME}". It is only meant for the following experiments: '
                     f'{", ".join(config.VALID_EXPERIMENT_NAMES)}.')
        sys.exit(1)

    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Generate abstractive summaries, settings are defined in the config.py file
    summary_generator = SummaryGenerator(config=config)
    summary_generator.generate_summaries()

# pylint: enable=logging-fstring-interpolation

if __name__ == "__main__":
    main()
