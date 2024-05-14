'''
auto_utils.py 
This script provides utility functions for generating summaries using any model from the 
AutoModelForCausalLM class in the transformers library and tokenizer from the AutoTokenizer class.
'''

# External imports
import os
import sys
import logging
from typing import Optional, List, Dict, Union
import torch
from torch.utils.data import DataLoader
from gensim.models.ldamodel import LdaModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from utils.evaluation_utils import get_topic_words
from config import get_generation_config, GENERATION_CONFIG, DATASET_CONFIG, EXPERIMENT_CONFIG,\
PROMPT_ENGINEERING_DICT, BASELINE_DICT, TOPICS_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_prompt(article: str, lda=None, tid: Optional[int] = None) -> str:
    """
    Constructs a prompt for model, optionally with a focus on specific topic words.
    
    :param article: The article text to be summarized.
    :param lda: The LDA model used to determine topic focus, if any.
    :param topic_id: The topic identifier for the focus topic, if any.
    :param num_topic_words: The number of top words from the topic to include in the prompt, if any.
    :param model_alias: The alias of the model used to generate summaries.
    :return: A string containing the structured prompt for summary generation.
    """
    num_topic_words = TOPICS_CONFIG['num_topic_words']
    model_alias = EXPERIMENT_CONFIG['model_alias']

    try:
        initial_instruction = "Generate a summary of the text" # "Summarize the text"
        if lda and tid and num_topic_words:
            top_words = get_topic_words(lda=lda, tid=tid, num_topic_words=num_topic_words)
            topic_description = ", ".join(top_words)
            topical_instruction = f" focussing on {topic_description}."
        else:
            topical_instruction = "."
        prompt = f'{initial_instruction}{topical_instruction}\n"{article}"\n'
        # "\nAs stated previously, {initial_instruction.lower()}{topical_instruction}'
        if model_alias == 'mistral_7b':
            prompt = f"<s>[INST] {prompt} [/INST]"
        return prompt
    except Exception as e:
        logger.error(f"Error generating prompt_engineering prompts: {e}")
        raise

def generate_prompt_engineering_summary(batch: Dict, index: int, lda: LdaModel,
                                        tokenizer: AutoTokenizer,
                                        device, model: AutoModelForCausalLM):
    '''
    Processes a batch of articles to generate summaries using different prompts.
    '''
    article, article_idx = batch['article'][index], batch['article_idx'][index].item()
    tid1, tid2 = batch['tid1'][index].item(), batch['tid2'][index].item()
    article_summaries = {
        'artciel_idx': article_idx,
        'tid1': tid1,
        'tid2': tid2,
        }

    focus_types = PROMPT_ENGINEERING_DICT['focus_types']
    model_alias = EXPERIMENT_CONFIG['model_alias']
    generation_config = get_generation_config(model_alias=model_alias, tokenizer=tokenizer)
    if model_alias in ['openelm_270m', 'openelm_450m', 'openelm_1b', 'openelm_3b']:
        max_length = 2048
    else:
        max_length = model.config.max_position_embeddings

    for focus_type in focus_types:
        tid = tid1 if focus_type == 'tid1_focus' else (tid2 if focus_type == 'tid2_focus' else None)
        prompt = generate_prompt(article, lda=lda if tid is not None else None, tid=tid)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                     max_length=max_length).to(device)
        outputs = model.generate(input_ids=tokenized_prompt['input_ids'],
                                 attention_mask=tokenized_prompt['attention_mask'],
                                 **generation_config.to_dict()
                                )
        decoded_summary = tokenizer.decode(outputs[:, tokenized_prompt['input_ids'].shape[1]:].squeeze(),
                                           skip_special_tokens=False)
        article_summaries[focus_type] = decoded_summary

    return article_summaries

def generate_prompt_engineering_summaries(dataloader: DataLoader, model: AutoModelForCausalLM,
                                          tokenizer: AutoTokenizer, device: torch.device,
                                          lda: Optional[LdaModel]
                                          ) -> List[Dict[str, Union[int, str]]]:
    """
    Generates topical abstractive summaries for a set of articles using different prompts.
    """
    try:
        model_alias = EXPERIMENT_CONFIG['model_alias']
        logger.info(f"Started generating summaries using {model_alias} model.")

        experiment_information = {
            'EXPERIMENT_CONFIG': EXPERIMENT_CONFIG,
            'GENERATION_CONFIG': GENERATION_CONFIG,
            'DATASET_CONFIG': DATASET_CONFIG,
            'TOPICS_CONFIG': TOPICS_CONFIG,
            'BASELINE_DICT': PROMPT_ENGINEERING_DICT,
        }
        results = {'experiment_information': experiment_information}
        generated_summaries = []

        num_articles = DATASET_CONFIG['num_articles']
        batch_size = DATASET_CONFIG['batch_size']

        model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():  # Operations inside don't track gradients
            for i, batch in enumerate(dataloader):
                if i * batch_size >= num_articles:
                    break

                for index in range(len(batch['article'])):
                    generated_summaries.append(
                        generate_baseline_summary(batch=batch, index=index, tokenizer=tokenizer,
                                                  device=device, model=model))

        logger.info(f"Generated {len(generated_summaries)} summaries.")
        results['generated_summaries'] = generated_summaries
        return results
    except Exception as e:
        logger.error(f"Error generating prompt_engineering summaries: {e}")
        raise

def generate_baseline_summary(batch: Dict, index: int, tokenizer: AutoTokenizer, device,
                              model: AutoModelForCausalLM):
    '''
    Generate one summary for a single article.
    '''
    article, article_idx = batch['article'][index], batch['article_idx'][index].item()
    article_summaries = {'artciel_idx': article_idx}

    model_alias = EXPERIMENT_CONFIG['model_alias']
    generation_config = get_generation_config(model_alias=model_alias, tokenizer=tokenizer)
    if model_alias in ['openelm_270m', 'openelm_450m', 'openelm_1b', 'openelm_3b']:
        max_length = 2048
    else:
        max_length = model.config.max_position_embeddings

    prompt = generate_prompt(article, )
    tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                 max_length=max_length).to(device)
    outputs = model.generate(input_ids=tokenized_prompt['input_ids'],
                             attention_mask=tokenized_prompt['attention_mask'],
                             **generation_config.to_dict())
    decoded_summary = tokenizer.decode(outputs[:, tokenized_prompt['input_ids'].shape[1]:].squeeze(),
                                       skip_special_tokens=False)
    article_summaries['summary'] = decoded_summary
    return article_summaries

def generate_baseline_summaries(dataloader: DataLoader, model: AutoModelForCausalLM,
                                tokenizer: AutoTokenizer, device: torch.device
                                )-> List[Dict[str, Union[int, str]]]:
    """
    Generates topical abstractive summaries for a set of articles.
    """
    try:
        model_alias = EXPERIMENT_CONFIG['model_alias']
        logger.info(f"Started generating summaries using {model_alias} model.")

        experiment_information = {
            'EXPERIMENT_CONFIG': EXPERIMENT_CONFIG,
            'GENERATION_CONFIG': GENERATION_CONFIG,
            'DATASET_CONFIG': DATASET_CONFIG,
            'TOPICS_CONFIG': TOPICS_CONFIG,
            'PLAIN_SUMMARISATION_DICT': BASELINE_DICT,
        }
        results = {'experiment_information': experiment_information}
        generated_summaries = []

        num_articles = DATASET_CONFIG['num_articles']
        batch_size = DATASET_CONFIG['batch_size']

        model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():  # Operations inside don't track gradients
            for i, batch in enumerate(dataloader):
                if i * batch_size >= num_articles:
                    break

                for index in range(len(batch['article'])):
                    generated_summaries.append(
                        generate_baseline_summary(batch=batch, index=index, tokenizer=tokenizer,
                                                  device=device, model=model))

        logger.info(f"Generated {len(generated_summaries)} summaries.")
        results['generated_summaries'] = generated_summaries
        return results
    except Exception as e:
        logger.error(f"Error generating plain_summarisation summaries: {e}")
        raise
