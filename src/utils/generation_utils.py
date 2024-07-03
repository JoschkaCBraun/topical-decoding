'''
generation_utils.py 
This script provides utility functions for generating summaries using any model from the 
AutoModelForCausalLM class in the transformers library and tokenizer from the AutoTokenizer class.
'''

# Standard library imports
import os
import sys
import json
import logging
from typing import Optional, List, Dict, Union

# Third-party imports
import torch
from gensim.models.ldamodel import LdaModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from config.experiment_config import Config
from src.logits_reweighting.logits_reweighting import CustomModel
from utils.load_and_get_utils import get_topic_vector, get_topic_words, load_model_and_tokenizer,\
    get_dataloader, load_lda, find_data_dir, get_topic_tokens

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_summaries()-> List[Dict[str, Union[int, str]]]:
    """
    Generates topical abstractive summaries for a set of articles.
    """
    try:
        experiment_name = Config.EXPERIMENT_NAME
        logger.info(f"Started generating summaries using {Config.MODEL_ALIAS} model.")

        experiment_information = {
            'EXPERIMENT_CONFIG': Config.EXPERIMENT_CONFIG,
            'GENERATION_CONFIG': Config.GENERATION_CONFIG,
            'DATASET_CONFIG': Config.DATASET_CONFIG,
            'TOPICS_CONFIG': Config.TOPICS_CONFIG,
        }

        lda = None
        dataloader = get_dataloader(dataset_name=Config.TEST_DATASET_NAME)
        if experiment_name in ['factor_scaling', 'constant_shift', 'threshold_selection']:
            tokenizer, model, device = load_model_and_tokenizer(CustomModel=CustomModel)
            model.model.eval() # Set the custom model to evaluation mode
        else:
            tokenizer, model, device = load_model_and_tokenizer()
            model.eval()

        if experiment_name == 'baseline':
            experiment_information['BASELINE_CONFIG'] = Config.BASELINE_CONFIG
        elif experiment_name == 'prompt_engineering':
            experiment_information['PROMPT_ENGINEERING_CONFIG'] = Config.PROMPT_ENGINEERING_CONFIG
            lda = load_lda()
        elif experiment_name == 'factor_scaling':
            experiment_information['FACTOR_SCALING_CONFIG'] = Config.FACTOR_SCALING_CONFIG
            factors = Config.FACTOR_SCALING_CONFIG['scaling_factors']
        elif experiment_name == 'constant_shift':
            experiment_information['CONSTANT_SHIFT_CONFIG'] = Config.CONSTANT_SHIFT_CONFIG
            factors = Config.CONSTANT_SHIFT_CONFIG['shift_constants']
        elif experiment_name == 'threshold_selection':
            experiment_information['THRESHOLD_SELECTION_CONFIG'] = Config.THRESHOLD_SELECTION_CONFIG
            factors = Config.THRESHOLD_SELECTION_CONFIG['selection_thresholds']
            topical_encouragement = Config.THRESHOLD_SELECTION_CONFIG['topical_encouragement']
        elif experiment_name == 'topic_vectors':
            experiment_information['TOPIC_VECTORS_CONFIG'] = Config.TOPIC_VECTORS_CONFIG
        else: 
            logger.error(f"Invalid experiment_name: {experiment_name} set in config.py.")
            raise ValueError(f"Invalid experiment_name: {experiment_name} set in config.py.")

        results = {'experiment_information': experiment_information}
        generated_summaries = []

        num_articles = Config.NUM_ARTICLES
        batch_size = Config.BATCH_SIZE

        with torch.no_grad():  # Operations inside don't track gradients
            for i, batch in enumerate(dataloader):
                if i * batch_size >= num_articles:
                    break

                for index in range(len(batch['article'])):
                    generated_summaries.append(
                        generate_individual_summary(batch=batch, index=index, tokenizer=tokenizer,
                                                    device=device, model=model, lda=lda))

        logger.info(f"Generated {len(generated_summaries)} summaries.")
        results['generated_summaries'] = generated_summaries
        store_results(results)
        return results
    except Exception as e:
        logger.error(f"Error generating {experiment_name} summaries: {e}")
        raise

def generate_individual_summary(batch: Dict, index: int, tokenizer: AutoTokenizer, device,
                                model: AutoModelForCausalLM, lda: Optional[LdaModel] = None,
                                ) -> Dict[str, Union[int, str]]:
    '''
    Generate summaries for a single article.
    '''

    experiment_name = Config.EXPERIMENT_CONFIG
    article, article_idx = batch['article'][index], batch['article_idx'][index].item()
    tid1 = batch['tid1'][index].item()
    article_summaries = {'artciel_idx': article_idx,
                         'tid1': tid1}
    model_alias = Config.MODEL_ALIAS

    generation_config = Config.get_generation_config(model_alias=model_alias, tokenizer=tokenizer)
    if model_alias in ['openelm_270m', 'openelm_450m', 'openelm_1b', 'openelm_3b']:
        max_length = 2048
    else:
        max_length = model.config.max_position_embeddings

    if experiment_name in ['baseline', 'topic_vectors']:
        prompt = generate_prompt(article)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                     max_length=max_length).to(device)
        if experiment_name == 'baseline':
            outputs = model.generate(input_ids=tokenized_prompt['input_ids'],
                                    attention_mask=tokenized_prompt['attention_mask'],
                                    **generation_config.to_dict())
        elif experiment_name == 'topic_vectors':

            for topic_encoding_type in Config.TOPIC_VECTORS_CONFIG:
                steering_vector = get_topic_vector(tid=tid1, topic_encoding_type=topic_encoding_type)
                with steering_vector.apply(model):
                    outputs = model.generate(input_ids=tokenized_prompt['input_ids'],
                                            attention_mask=tokenized_prompt['attention_mask'],
                                            **generation_config.to_dict())
                decoded_summary = tokenizer.decode(outputs[:, tokenized_prompt['input_ids'].shape[1]:].squeeze(),
                                           skip_special_tokens=False)
                article_summaries[topic_encoding_type] = decoded_summary

    elif experiment_name == 'prompt_engineering':
        tid1, tid2 = batch['tid1'][index].item(), batch['tid2'][index].item()
        article_summaries['tid1'] = tid1
        article_summaries['tid2'] = tid2
        focus_types = Config.PROMPT_ENGINEERING_CONFIG['focus_types']

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

    elif experiment_name in ['factor_scaling', 'constant_shift', 'threshold_selection']:
        topic_tokens = get_topic_tokens(tokenizer=tokenizer, lda=lda, tid=tid)

        prompt = generate_prompt(article=article)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                     max_length=max_length).to(device)

        # Set parameters for generation as defined in the config.py file
        generation_config = Config.get_generation_config(model_alias=model_alias,
                                                         tokenizer=tokenizer)
        topical_encouragement = None
        if experiment_name == 'factor_scaling':
            factors = Config.FACTOR_SCALING_CONFIG['scaling_factors']
        elif experiment_name == 'constant_shift':
            factors = Config.CONSTANT_SHIFT_CONFIG['shift_constants']
        elif experiment_name == 'threshold_selection':
            factors = Config.THRESHOLD_SELECTION_CONFIG['selection_thresholds']
            topical_encouragement = Config.THRESHOLD_SELECTION_CONFIG['topical_encouragement']

        for factor in factors:
            output = model.generate(
                experiment_name=experiment_name,
                input_ids=tokenized_prompt['input_ids'],
                attention_mask=tokenized_prompt['attention_mask'],
                topic_token_ids=topic_tokens,
                scaling_factor=factor,
                shift_constant=factor,
                selection_threshold=factor,
                topical_encouragement=topical_encouragement,
                **generation_config.to_dict())

            # summary = tokenizer.decode(output[tokenized_prompt['input_ids'].shape[1]:], skip_special_tokens=False)
            decoded_summary = tokenizer.decode(output.squeeze(), skip_special_tokens=False)
            article_summaries[str(factor)] = decoded_summary

    return article_summaries

def generate_prompt(article: str, lda=None, tid: Optional[int] = None) -> str:
    """
    Constructs a prompt for model, optionally with a focus on specific topic words.
    
    :param article: The article text to be summarized.
    :param lda: The LDA model used to determine topic focus, if any.
    :param tid: The topic identifier for the focus topic, if any.
    :return: A string containing the structured prompt for summary generation.
    """
    num_topic_words = Config.NUM_TOPIC_WORDS
    model_alias = Config.MODEL_ALIAS

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

def store_results(summaries: List[Dict[str, str]]) -> None:
    """
    Stores the generated summaries to a file in the results directory.

    :param summaries: The generated summaries to store.
    """

    file_name = (f"{Config.EXPERIMENT_NAME}_{Config.MODEL_ALIAS}_{Config.NUM_ARTICLES}_"
                 f"{Config.MIN_NEW_TOKENS}_{Config.MAX_NEW_TOKENS}_{Config.NUM_BEAMS}.json")
    start_path = os.getcwd()
    data_path = find_data_dir(start_path)
    if not data_path:
        logging.error("Data directory not found. Summaries not stored.")
        return

    results_dir = os.path.join(data_path, 'results_{experiment_name}', file_name)
    with open(results_dir, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=4)
    logging.info("Summaries stored.")
    return
