"""
Creates a list of dictionaries, each containing a document and its associated topic words and a summary. 
The list is stored in init_examples.json. This list is used to initialize the LlamaModelHandler in the main function.

"""
import json
import logging
import os
import sys
from typing import Dict, List

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

# Local imports (adjust these as per your project's structure)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from utils.newts_lda_utils import get_topic_words, read_LDA, read_NEWTS_train

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

N_TOPIC_WORDS = 20
N_ARTICLES = 20


def main() -> None:
    try:
        # Set up directories
        data_dir = os.path.join(parent_dir, "data")
        outputs_dir = os.path.join(data_dir, "self_refine_data")
        os.makedirs(outputs_dir, exist_ok=True)

        # Load the NEWTS dataset
        train = read_NEWTS_train(os.path.join(data_dir, "NEWTS/NEWTS_train_2400.csv"))

        # Load LDA model and dictionary
        lda, dictionary = read_LDA(os.path.join(data_dir, "LDA_250"))

        # create list of dictionaries (document and topic words)
        doc_topic_list: List[Dict[str, List[str]]] = []
        for i in range(min(N_ARTICLES, len(train))):
            document = train.iloc[i]["article"]
            tid1 = train.iloc[i]["tid1"]
            tid2 = train.iloc[i]["tid2"]
            summary1 = train.iloc[i]["summary1"]
            summary2 = train.iloc[i]["summary2"]

            for topic_id in [tid1, tid2]:
                topic_words = get_topic_words(lda, topic_id, N_TOPIC_WORDS)
                doc_topic_dict = {
                    "document": document,
                    "topic_words": topic_words,
                    "summary": summary1 if topic_id == tid1 else summary2,
                }
                doc_topic_list.append(doc_topic_dict)

        # Save the doc_topic_list as JSON
        with open(os.path.join(outputs_dir, "init_examples_NEWTS.json"), "w") as f:
            json.dump(doc_topic_list, f, indent=4)

        logging.info("Document topic list has been successfully saved.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
