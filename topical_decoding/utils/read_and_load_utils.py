"""
Utility functions for loading models from the transformers AutoModelForCausalLM class and tokenizers
from AutoTokenizer class. Also utility functions for loading the NEWTS dataset, LDA model and 
dictionary.
"""
# Standard library imports
import os
import sys
import logging
from typing import Dict, Optional, Tuple

# Third-party imports
import torch
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from config import DATASET_CONFIG, EXPERIMENT_CONFIG, MODEL_ALIAS_DICT

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(CustomModel: Optional[AutoModelForCausalLM] = None,
                             device_map: str = "auto",
                             ) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """
    Loads the model and tokenizer for a given model alias.

    :param CustomModel: The custom model to be loaded, if any.
    :param device_map: The device on which the model is to be loaded.
    :return: A tuple containing the tokenizer, model, and device.
    """
    try:
        model_alias = EXPERIMENT_CONFIG['model_alias']
        model_details = MODEL_ALIAS_DICT.get(model_alias, {})
        model_name = model_details.get('model_name')
        tokenizer_name = model_details.get('tokenizer_name')
        hf_auth_token = model_details.get('hf_auth_token')
        logger.info(f"Loading model: {model_name} and tokenizer: {tokenizer_name}")

        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_name,
                                                  padding_side='left',
                                                  torch_dtype=torch.float16,
                                                  token=hf_auth_token)
        
        experiment_name = EXPERIMENT_CONFIG['experiment_name']
        if experiment_name in ['factor_scaling', 'constant_shift', 'threshold_selection']:
            if CustomModel:
                model = CustomModel(pretrained_model_name_or_path=model_name,
                                    device_map=device_map,
                                    torch_dtype=torch.float16,
                                    token=hf_auth_token,
                                    trust_remote_code=True)
                model.model.to(device)
                model.model.generation_config.pad_token_id = tokenizer.eos_token_id
            else:
                logger.error("Custom model is required for this experiment: {experiment_name}.")
                raise ValueError(f"Custom model is required for this experiment: {experiment_name}.")
        else:
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                         device_map=device_map,
                                                         torch_dtype=torch.float16,
                                                         token=hf_auth_token,
                                                         trust_remote_code=True).to(device)
            model.generation_config.pad_token_id = tokenizer.eos_token_id

        # Ensure the tokenizer has a pad token, set to eos token if not
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if CustomModel:
                model.model.resize_token_embeddings(len(tokenizer))
            else:
               model.resize_token_embeddings(len(tokenizer))

        # Gemma's activation function should be approximate GeLU and not exact GeLU.
        # Changing the activation function to `gelu_pytorch_tanh`.if you want to use the legacy
        # `gelu`, edit the `model.config` to set `hidden_activation=gelu` instead of `hidden_act`.
        # See https://github.com/huggingface/transformers/pull/29402 for more details.
        if model_name in ['google/gemma-2b-it', 'google/gemma-7b-it']:
            if CustomModel:
                previous_activation = model.model.config.hidden_act
                model.model.config.hidden_act = 'hidden_act'
                logger.info(f"Changed activation function to 'hidden_act' from "
                            f"{previous_activation} for model: {model_name}")
            else:
                previous_activation = model.config.hidden_act
                model.config.hidden_act = 'hidden_act'
                logger.info(f"Changed activation function to 'hidden_act' from "
                            f"{previous_activation} for model: {model_name}")
        logger.info(f"Loaded model: {model_name} and tokenizer: {tokenizer_name}")

    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise
    return tokenizer, model, device

def load_tokenizer(model_alias: str) -> AutoTokenizer:
    """
    Loads and returns the tokenizer for a given model alias. Used for evaluation purposes.
    
    :return: The tokenizer for the specified model.
    """
    try:
        model_details = MODEL_ALIAS_DICT.get(model_alias, {})
        tokenizer_name = model_details.get('tokenizer_name')
        hf_auth_token = model_details.get('hf_auth_token')
        logger.info(f"Loading tokenizer: {tokenizer_name}")

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_name,
                                                  token=hf_auth_token)
        logger.info(f"Loaded tokenizer: {tokenizer_name}")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

def find_project_root(start_path):
    """
    Recursively finds and returns the path to the project root, which is indicated by the presence of a '.git' directory.
    """
    current_path = start_path
    while current_path != os.path.dirname(current_path):  # Iterate until the filesystem root is reached
        if os.path.exists(os.path.join(current_path, '.git')):
            return current_path
        current_path = os.path.dirname(current_path)
    return None  # Return None if no project root is found

def find_data_dir(start_path):
    """
    Dynamically finds and returns the path to the 'data' directory starting from a given path.
    It searches for the project root first and then for the 'data' directory.
    """
    project_root = find_project_root(start_path)
    if project_root is None:
        logging.error("Project root not found.")
        return None

    potential_data_path = os.path.join(project_root, 'topical_decoding', 'data')
    if os.path.exists(potential_data_path):
        logging.info(f"Data directory found at {potential_data_path}.")
        return potential_data_path
    else:
        logging.error("Data directory not found in the project structure.")
        return None

def read_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Loads and returns the specified dataset as a Pandas DataFrame.
    
    :param dataset_name: The name of the dataset to load.
    :return: The specified dataset DataFrame or None if an error occurs.
    """
    start_path = os.getcwd()
    data_path = find_data_dir(start_path)
    if not data_path:
        return None
    if dataset_name == 'newts_train':
        dataset_name = 'NEWTS_train_2400'
    elif dataset_name == 'newts_test':
        dataset_name = 'NEWTS_test_600'
    else:
        logging.error(f"Given dataset_name {dataset_name} is not valid. Please provide a valid"/
                      "dataset name with 'newts_train' or 'newts_test'")
        return None
    file_path = os.path.join(data_path, "NEWTS", f"{dataset_name}.csv")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df.rename(columns={df.columns[0]: 'article_idx'}, inplace=True)
        logging.info(f"Successfully loaded {dataset_name} dataset.")
        return df
    except Exception as e:
        logging.error(f"Error reading {dataset_name} dataset: {e}")
        return None

class NEWTSDataset(Dataset):
    """
    Dataset class for the NEWTS dataset.
    """
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        article = self.data.iloc[idx]
        return {
            'article': article['article'],
            'summary1': article['summary1'],
            'summary2': article['summary2'],
            'article_idx': article['article_idx'],
            'tid1': article['tid1'],
            'tid2': article['tid2'],
        }

def setup_dataloader(dataset_name: str) -> DataLoader:
    """
    Sets up and returns a DataLoader for either the NEWTS training or testing set.

    :param dataset_name: The name of the dataset to set up the DataLoader for.
    :return: A DataLoader for the specified NEWTS dataset.
    """
    try:
        dataloader = DataLoader(dataset=NEWTSDataset(dataframe=read_dataset(dataset_name)),
                                batch_size=DATASET_CONFIG['batch_size'],
                                num_workers=DATASET_CONFIG['num_workers'],
                                shuffle=DATASET_CONFIG['shuffle'])
        return dataloader
    except Exception as e:
        logging.error(f"Error setting up the DataLoader: {e}")
        return None

def load_lda_and_dictionary(should_load_lda: bool=True, should_load_dictionary: bool=True) -> Tuple:
    """
    Loads and returns the LDA model and dictionary from the filesystem.
    
    Returns:
    - tuple: (LdaModel or None, Dictionary or None) depending on success or failure of loading.
    """
    start_path = os.getcwd()
    data_path = find_data_dir(start_path)
    if not data_path:
        return None, None

    lda_model_path = os.path.join(data_path, "LDA_250", 'lda.model')
    dictionary_path = os.path.join(data_path, "LDA_250", 'dictionary.dic')
    lda, dictionary = None, None

    try:
        if should_load_lda:
            lda = LdaModel.load(lda_model_path, mmap='r')
            logging.info("Successfully loaded the LDA model.")
        if should_load_dictionary:
            dictionary = Dictionary.load(dictionary_path, mmap='r')
            logging.info("Successfully loaded the dictionary.")
        return lda, dictionary

    except Exception as e:
        logging.error(f"Error loading the LDA model or dictionary: {e}")
        return None, None

def load_dictionary() -> Dictionary:
    """Loads and returns the dictionary from the filesystem."""
    _, dictionary = load_lda_and_dictionary(should_load_lda=False, should_load_dictionary=True)
    return dictionary

def load_lda() -> LdaModel:
    """
    Loads and returns the LDA model from the filesystem.
    """
    lda, _ = load_lda_and_dictionary(should_load_lda=True, should_load_dictionary=False)
    return lda
