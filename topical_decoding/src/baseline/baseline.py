"""
baseline.py

This script generates one abstractive summary for each article of the NEWTS dataset
using any model from the AutoModelForCausalLM class in the Hugging Face transformers
library and tokenizer from the AutoTokenizer class. 
Results are stored in JSON files in the data/results_baseline folder.
"""

# Standard library imports
import os
import sys
import json
import logging

# Third-party imports
import torch

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from config import DATASET_CONFIG, GENERATION_CONFIG, EXPERIMENT_CONFIG
from utils.generation_utils import generate_summaries

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """ Generate plain abstractive summaries of articles from the NEWTS dataset."""

    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    experiment_name = EXPERIMENT_CONFIG['experiment_name']
    if experiment_name != 'baseline':
        logging.error("This script is meant for the baseline experiment only.")
        sys.exit(1)

    # Generate summaries
    summaries = generate_summaries()

    # Save summaries to a file in the results_baseline directory
    file_name = (f"{experiment_name}_{EXPERIMENT_CONFIG['model_alias']}_"
                 f"{DATASET_CONFIG['num_articles']}_{GENERATION_CONFIG['min_new_tokens']}_"
                 f"{GENERATION_CONFIG['max_new_tokens']}_{GENERATION_CONFIG['num_beams']}.json")
    results_dir = os.path.join(parent_dir, 'data', 'results_baseline', file_name)
    with open(results_dir, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=4)
    logging.info("Summaries stored.")

if __name__ == "__main__":
    main()
