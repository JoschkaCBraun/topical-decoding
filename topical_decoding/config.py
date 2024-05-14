"""
config.py
This script contains all the hyperparameters that can be adapted by the user.
"""

# Third-party imports
from transformers import GenerationConfig, AutoTokenizer

# Choose experiment between 'baseline', 'prompt_engineering', 'constant_shift', 'factor_scaling' and 'threshold_selection'
EXPERIMENT_NAME = 'baseline'

# Choose model between 'openelm_270m', 'openelm_450m', 'openelm_1b', 'openelm_3b', 'gemma_2b', 'gemma_7b', 'falcon_7b', 'mistral_7b', 'llama_8b'
MODEL_ALIAS = 'openelm_270m'

EXPERIMENT_CONFIG = {
    'experiment_name': EXPERIMENT_NAME,
    'model_alias': MODEL_ALIAS,
}

# Experiment agnostic hyperparameters
DATASET_NAME = 'newts_train' # choose from 'newts_train', 'newts_test'
NUM_ARTICLES = 3  # number of articles for which summaries are generated
BATCH_SIZE = 1  # batch size for data loader ! Code is not prepared for batch size > 1 !
NUM_WORKERS = 0  # number of workers for generating summaries
SHUFFLE = False  # shuffle data loader

DATASET_CONFIG = {
    'dataset_name': DATASET_NAME,
    'num_articles': NUM_ARTICLES,
    'batch_size': BATCH_SIZE,
    'num_workers': NUM_WORKERS,
    'shuffle': SHUFFLE,
}

# Topic model hyperparameters
NUM_TOPIC_WORDS = 25  # number of words from topic model to use as features
MIN_PHI_VALUE = 0.001  # minimum value of phi to consider when extracting topic words

TOPICS_CONFIG = {
    'num_topic_words': NUM_TOPIC_WORDS,
    'min_phi_value': MIN_PHI_VALUE,
}

# GenerationConfig hyperparameters
MIN_NEW_TOKENS = 80 # minimum number of tokens in the generated summary
MAX_NEW_TOKENS = 90 # maximum number of tokens in the generated summary
NUM_BEAMS = 1 # number of beams for beam search
DO_SAMPLE = True # True for sampling, False for greedy decoding
TOP_P = 0.95 # nucleus sampling parameter
TOP_K = 50 # top-k sampling parameter
# EARLY_STOPPING = True
# decoder_start_token_id=0,#
# repetition_penalty=2.0,  # Penalize immediate repetitions
# no_repeat_ngram_size=3,  # Prevent repeating any 2-gram
# eos_token_id=model.config.eos_token_id,
# pad_token=model.config.pad_token_id,
# num_return_sequences=1
# low_memory=True,
# pad_token_id=tokenizer.pad_token_id,
# temperature=1.0,

GENERATION_CONFIG = {
    'min_new_tokens': MIN_NEW_TOKENS,
    'max_new_tokens': MAX_NEW_TOKENS,
    'num_beams': NUM_BEAMS,
    'do_sample': DO_SAMPLE,
    'top_p': TOP_P,
    'top_k': TOP_K,
}

# Experiment specific hyperparameters
BASELINE_DICT = {

}

PROMPT_ENGINEERING_DICT = {
    'focus_types': ['tid1_focus', 'no_focus', 'tid2_focus'] # focus types for prompts
}

FACTOR_SCALING_DICT = {
    'scaling_factors': [0.125 ,0.25, 0.5, 1, 2, 4, 8]  # factors for scaling logits
}

CONSTANT_SHIFT_DICT = {
    'shift_constants': [-5, -2, -1, 0, 1, 2, 5]  # values for shifting logits
}

THRESHOLD_SELECTION_DICT = {
    'selection_thresholds': [1.0, 0.2, 0.05, 0.01, 0.005], # thresholds for selecting topic words
    'topical_encouragement': 0.0 # factor for encouraging topic words
}

def get_generation_config(model_alias: str, tokenizer: AutoTokenizer) -> GenerationConfig:
    '''Get the GenerationConfig for the model alias.
    Args:
        model_alias (str): Alias of the model.
    Returns:
        GenerationConfig: GenerationConfig for the model.'''
    if model_alias not in MODEL_ALIAS_DICT:
        raise ValueError(f"Model alias {model_alias} not found in config/model_configurations.")

    model_name = MODEL_ALIAS_DICT[model_alias]['model_name']
    generation_config = GenerationConfig.from_pretrained(model_name,
                                                         max_new_tokens=MAX_NEW_TOKENS,
                                                         min_new_tokens=MIN_NEW_TOKENS,
                                                         num_beams=NUM_BEAMS,
                                                         do_sample=DO_SAMPLE,
                                                         top_p=TOP_P,
                                                         top_k=TOP_K,
                                                         max_length=None,
                                                         min_length=None,
                                                         pad_token_id=tokenizer.eos_token_id,
                                                         use_cache=True)
    return generation_config

MODEL_ALIAS_DICT= {
     'openelm_270m': {
        'model_name': "apple/OpenELM-270M-Instruct",
        'tokenizer_name': "meta-llama/Llama-2-7b-hf",
        'hf_auth_token': "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"
    },
    'openelm_450m': {
        'model_name': "apple/OpenELM-450M-Instruct",
        'tokenizer_name': "meta-llama/Llama-2-7b-hf",
        'hf_auth_token': "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"
    },
    'openelm_1b': {
        'model_name': "apple/OpenELM-1_1B-Instruct",
        'tokenizer_name': "meta-llama/Llama-2-7b-hf",
        'hf_auth_token': "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"
    },
    'openelm_3b': {
        'model_name': "apple/OpenELM-3B-Instruct",
        'tokenizer_name': "meta-llama/Llama-2-7b-hf",
        'hf_auth_token': "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"
    },   
    'gemma_2b': {
        'model_name': "google/gemma-2b-it",
        'tokenizer_name': "google/gemma-2b-it",
        'hf_auth_token': "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"
    },
    'gemma_7b': {
        'model_name': 'google/gemma-7b-it',
        'tokenizer_name': 'google/gemma-7b-it',
        'hf_auth_token': "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"
    },
    'mistral_7b': {
        'model_name': "mistralai/Mistral-7B-Instruct-v0.1",
        'tokenizer_name': "mistralai/Mistral-7B-Instruct-v0.1",
        'hf_auth_token': None
    },
    'falcon_7b': {
        'model_name': "tiiuae/falcon-7b-instruct",
        'tokenizer_name': "tiiuae/falcon-7b-instruct",
        'hf_auth_token': None
    },
    'llama_8b': {
        'model_name': "meta-llama/Meta-Llama-3-8B-Instruct",
        'tokenizer_name': "meta-llama/Meta-Llama-3-8B-Instruct",
        'hf_auth_token': "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"
        }
}

ROUGE_METRICS = ['rouge1', 'rouge2', 'rougeL']