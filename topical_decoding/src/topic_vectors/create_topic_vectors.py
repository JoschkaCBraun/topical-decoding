"""
create_topic_vectors.py

This script creates a topic vector for each topic.
"""
import logging

# Standard library imports
import os
import pickle
import sys
from typing import Dict, List, Tuple

# Third-party imports
import pandas as pd

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from config import (
    DATASET_CONFIG,
    EXPERIMENT_CONFIG,
    TOPIC_VECTORS_CONFIG,
    TOPICS_CONFIG,
)
from steering_vectors import train_steering_vector
from utils.evaluation_utils import get_topic_words
from utils.read_and_load_utils import (
    get_topic_vectors_folder_name,
    load_lda,
    load_model_and_tokenizer,
    read_dataset,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_samples_from_summaries(
    tid_positive: int, df: pd.DataFrame
) -> List[Tuple[str, str]]:
    """Create training samples from summaries to train the steering vector for tid_positive.

    :param tid_positive: The positive topic id for which the steering vector is to be trained.
    :param df: A pandas DataFrame containing the dataset.
    :return: A list of tuples of (positive_prompt, negative_prompt).
    """
    training_samples = []

    include_non_matching = TOPIC_VECTORS_CONFIG["include_non_matching"]
    num_samples = TOPIC_VECTORS_CONFIG["num_samples"]

    # Filter summaries specifically associated with the positive topic id
    positive_summaries = (
        df.loc[df["tid1"] == tid_positive, "summary1"].tolist()
        + df.loc[df["tid2"] == tid_positive, "summary2"].tolist()
    )

    len_positive_summaries = len(positive_summaries)
    if len_positive_summaries == 0:
        logger.error(f"No summaries found for topic id {tid_positive}")
        raise ValueError(f"No summaries found for topic id {tid_positive}")

    non_matching_samples_needed = max(0, num_samples - len_positive_summaries)
    pos_idx = 0

    for _, row in df.iterrows():
        if row["tid1"] == tid_positive:
            training_samples.append((row["summary1"], row["summary2"]))

        elif row["tid2"] == tid_positive:
            training_samples.append((row["summary2"], row["summary1"]))

        elif include_non_matching and non_matching_samples_needed > 0:
            training_samples.append(
                (positive_summaries[pos_idx % len_positive_summaries], row["summary1"])
            )
            non_matching_samples_needed -= 1
            if non_matching_samples_needed > 0:
                training_samples.append(
                    (
                        positive_summaries[(pos_idx + 1) % len_positive_summaries],
                        row["summary2"],
                    )
                )
                non_matching_samples_needed -= 1
            pos_idx += 2

    return training_samples


def generate_topic_strings(df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    """Generate topic strings for all relevant topic ids in the dataset.

    :param df: A pandas DataFrame containing the dataset.
    :param lda: A trained LDA model.
    :param num_topic_words: The number of words to include in the topic words.
    :return: A dictionary of topic strings for all relevant topic ids.
    """
    relevant_tids = set(map(int, df["tid1"].unique())).union(
        set(map(int, df["tid2"].unique()))
    )
    topic_strings = {tid: {} for tid in relevant_tids}
    lda = load_lda()
    num_topic_words = TOPICS_CONFIG["num_topic_words"]

    for tid in relevant_tids:
        topic_words = get_topic_words(lda=lda, tid=tid, num_topic_words=num_topic_words)
        topic_strings[tid]["topic_words"] = " ".join(topic_words)

    missing_tids = set(relevant_tids)
    for _, row in df.iterrows():
        tid1, tid2 = row["tid1"], row["tid2"]
        if tid1 in missing_tids:
            topic_strings[tid1]["topic_phrases"] = row["phrases1"]
            topic_strings[tid1]["topic_description"] = row["sentences1"]
            missing_tids.remove(tid1)
        if tid2 in missing_tids:
            topic_strings[tid2]["topic_phrases"] = row["phrases2"]
            topic_strings[tid2]["topic_description"] = row["sentences2"]
            missing_tids.remove(tid2)
        if len(missing_tids) == 0:
            break

    return topic_strings


def create_samples_from_strings(
    tid_positive: int,
    topic_strings: Dict[int, Dict[str, str]],
) -> List[Tuple[str, str]]:
    """Create training samples from topic strings to traing topic vector for tid_pos.

    :param tid_positive: The positive topic id for which the steering vector is to be trained.
    :param topic_strings: A dictionary of topic strings for all relevant topic ids.
    :return: A list of tuples of (positive_prompt, negative_prompt).
    """

    training_samples = []

    for tid_negative, negative_data in topic_strings.items():
        if tid_negative == tid_positive:
            continue
        for key, positive_data in topic_strings[tid_positive].items():
            training_samples.append((positive_data, negative_data[key]))

    return training_samples


def create_training_samples(
    topic_encoding_type: str,
) -> Dict[int, List[Tuple[str, str]]]:
    """Create training samples for the steering vector.

    :param topic_encoding_type: The type of topic encoding to use.
    :return: A dictionary of training samples for each topic.
    """
    df = read_dataset(dataset_name=DATASET_CONFIG["dataset_name"])
    training_samples = dict()

    if topic_encoding_type == "zeros":
        relevant_tids = list(map(int, df["tid1"].unique()))
        for tid in relevant_tids:
            training_samples[tid] = [("same text", "same text")]

    elif topic_encoding_type == "topic_strings":
        topic_strings = generate_topic_strings(df=df)
        for tid in topic_strings.keys():
            training_samples[tid] = create_samples_from_strings(
                tid_positive=tid, topic_strings=topic_strings
            )
    elif topic_encoding_type == "topical_summaries":
        relevant_tids = list(map(int, df["tid1"].unique()))
        for tid in relevant_tids:
            training_samples[tid] = create_samples_from_summaries(
                tid_positive=tid, df=df
            )
    else:
        logger.error(f"Invalid topic encoding type: {topic_encoding_type}")
        raise ValueError(f"Invalid topic encoding type: {topic_encoding_type}")

    return training_samples


def train_topic_vectors(topic_encoding_type: str) -> None:
    """Train the steering vectors for each topic in the topic strings.

    :param topic_encoding_type: The type of topic encoding to use.
    """

    tokenizer, model, _ = load_model_and_tokenizer(CustomModel=None)

    training_samples = create_training_samples(topic_encoding_type=topic_encoding_type)
    topic_counter = 0
    num_topics = len(training_samples)

    folder_name = get_topic_vectors_folder_name()
    results_dir = os.path.join(parent_dir, "data", "topic_vectors_data", folder_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Train steering vectors for each topic
    for tid, training_samples in training_samples.items():
        topic_vector = train_steering_vector(
            model=model,
            tokenizer=tokenizer,
            training_samples=training_samples,
            show_progress=True,
        )
        topic_vector_dir = os.path.join(results_dir, f"topic_vector_tid_{str(tid)}.pkl")
        with open(topic_vector_dir, "wb") as f:
            pickle.dump(topic_vector, f)
        # log how many topics have been processed
        topic_counter += 1
        logger.info(
            f"Saved topic vector for topic id {tid} to {topic_vector_dir} ({topic_counter}/{num_topics})"
        )
