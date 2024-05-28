'''
topic_vectors.py

This script tests the steering vectors for each topic in the topic strings.
'''
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
from config import EXPERIMENT_CONFIG, DATASET_CONFIG, GENERATION_CONFIG
from utils.generation_utils import generate_summaries

def main():
    '''Generate summaries with topic vectors for steering the model to a specific topic.'''

    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    experiment_name = EXPERIMENT_CONFIG['experiment_name']
    if experiment_name != 'topic_vectors':
        logging.error("This script is meant for the topic_vectors experiment only.")
        sys.exit(1)

    summaries = generate_summaries()

    # Save summaries to a file in the results_topic_vectors directory
    file_name = (f"{experiment_name}_{EXPERIMENT_CONFIG['model_alias']}_"
                 f"{DATASET_CONFIG['num_articles']}_{GENERATION_CONFIG['min_new_tokens']}_"
                 f"{GENERATION_CONFIG['max_new_tokens']}_{GENERATION_CONFIG['num_beams']}.json")
    
    results_dir = os.path.join(parent_dir, 'data', 'results_topic_vectors', file_name)
    with open(results_dir, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=4)
    logging.info("Summaries stored.")
if __name__ == "__main__":
    main()