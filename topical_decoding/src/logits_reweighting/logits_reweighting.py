"""
logits_reweighting.py
This script generates abstractive summaries of articles from the NEWTS dataset using custom models
build on any model from the AutoModelForCausalLM class in the transformers library and tokenizer 
from the AutoTokenizer class. The custom models reweight token logits based on their relevance to
a specified topic. Results are stored in JSON files.
"""

import json
import logging

# Standard library imports
import os
import sys
from typing import Dict, List, Optional, Union

# Third-party imports
import torch
from gensim.models.ldamodel import LdaModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from config import (
    CONSTANT_SHIFT_DICT,
    DATASET_CONFIG,
    EXPERIMENT_CONFIG,
    FACTOR_SCALING_DICT,
    GENERATION_CONFIG,
    THRESHOLD_SELECTION_DICT,
    TOPICS_CONFIG,
    get_generation_config,
)
from utils.evaluation_utils import get_topic_tokens
from utils.generation_utils import generate_prompt
from utils.read_and_load_utils import (
    load_lda,
    load_model_and_tokenizer,
    setup_dataloader,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FactorScalingLogitsProcessor(LogitsProcessor):
    """
    A custom logits processor that adjusts the token logits based on their relevance to a specified
    topic, either encouraging or discouraging the model from generating these tokens.
    """

    def __init__(self, topic_token_ids: List[int], scaling_factor: float):
        """
        Initializes the processor with a set of token IDs to be adjusted and a factor by which their logits are multiplied.

        :param topic_token_ids: Iterable of token IDs that are to be adjusted.
        :param scaling_factor: The factor by which to multiply the logits of the specified tokens.
        """
        super().__init__()
        self.topic_token_ids = topic_token_ids
        self.scaling_factor = scaling_factor

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Scales the logits of tokens with topic_tokens_id with scaling_factor during the generation process.

        :param input_ids: Tensor of input IDs.
        :param scores: Tensor of logits for each token in the vocabulary.
        :return: Modified scores tensor with adjusted logits for specified tokens.
        """
        device = scores.device
        vocab_size = scores.shape[1]

        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for token_id in self.topic_token_ids:
            mask[token_id] = True

        scores[:, mask] *= self.scaling_factor
        return scores


class ConstantShiftLogitsProcessor(LogitsProcessor):
    """
    A custom logits processor that adjusts the token logits based on their relevance to a specified topic,
    either encouraging or discouraging the model from generating these tokens.
    """

    def __init__(self, topic_token_ids: List[int], shift_constant: float):
        """
        Initializes the processor with a set of token IDs to be adjusted and a factor by which their logits are multiplied.

        :param topic_token_ids: Iterable of token IDs that are to be adjusted.
        :param shift_constant: The constant value by which to shift the logits of the specified tokens.
        """
        super().__init__()
        self.topic_token_ids = topic_token_ids
        self.shift_constant = shift_constant

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Shifts the logits of tokens with topic_tokens_id by constant_shift during the generation process.

        :param input_ids: Tensor of input IDs.
        :param scores: Tensor of logits for each token in the vocabulary.
        :return: Modified scores tensor with adjusted logits for specified tokens.
        """
        device = scores.device
        vocab_size = scores.shape[1]

        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for token_id in self.topic_token_ids:
            mask[token_id] = True

        scores[:, mask] += self.shift_constant
        return scores


class ThresholdSelectionLogitsProcessor(LogitsProcessor):
    """
    A custom logits processor that adjusts the token logits of topic tokens based on their
    relevance to a specified topic.
    """

    def __init__(
        self, topic_token_ids, selection_threshold: float, topical_encouragement: float
    ):
        """
        Initializes the processor with a set of token IDs to be adjusted, a threshold for selecting
        topic tokens, and an encouragement factor for topic tokens.

        :param topic_token_ids: Iterable of token IDs that are to be adjusted.
        :param selection_threshold: The threshold for selecting topic tokens.
        :param topical_encouragement: The factor by which to adjust the logits of selected topic tokens.
        """
        super().__init__()
        self.topic_token_ids = topic_token_ids
        self.selection_threshold = selection_threshold
        self.topical_encouragement = topical_encouragement

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Ensure the new tensors are created on the same device as the input "scores"
        device = scores.device
        dtype = scores.dtype

        # Apply softmax to convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)

        # Print top 5 tokens with highest logits
        # topk_logits = torch.topk(scores, 5, dim=-1)
        # print(f"Top 5 token logits: {topk_logits[0]}")

        # Print top 5 tokens with highest probabilities
        # topk_probs = torch.topk(probs, 5, dim=-1)
        # print(f"Top 5 token probabilities: {topk_probs[0]}")

        # Initizalize a mask to identify the topic tokens in the logits
        topic_mask = torch.full(scores.shape, False, dtype=torch.bool, device=device)
        topic_mask[:, self.topic_token_ids] = True

        # Calculate probabilities for topic tokens
        topic_probs = torch.where(
            topic_mask, probs, torch.tensor(0.0, dtype=dtype, device=device)
        )

        # Identify topic tokens with probabilities above the threshold
        exceed_threshold_mask = topic_probs >= self.selection_threshold

        # Find the maximum logit value in each item of the batch
        max_logit, _ = torch.max(scores, dim=-1, keepdim=True)

        # Update logits for topic tokens largen than threshold to max_logit + topical_encouragement
        # We only update logits where exceed_threshold_mask is True
        scores = torch.where(
            exceed_threshold_mask, max_logit + self.topical_encouragement, scores
        )

        return scores


class CustomModel:
    """
    A model that uses a custom logits processor to reweight logits based on their topic relevance.
    """

    def __init__(self, pretrained_model_name_or_path, *args, **kwargs):
        # Load the underlying model instance
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

    def generate(
        self,
        input_ids: List[int],
        attention_mask=None,
        topic_token_ids=None,
        scaling_factor: Optional[float] = None,
        shift_constant: Optional[float] = None,
        selection_threshold: Optional[float] = None,
        topical_encouragement: Optional[float] = None,
        *args,
        **kwargs,
    ):
        """
        Generates text sequences, optionally adjusting token logits based on their topic relevance.

        :param input_ids: Tensor of input IDs.
        :param attention_mask: Optional tensor indicating which tokens should be attended to.
        :param topic_token_ids: List of token IDs that are relevant to the topic.
        :param scaling_factor: Factor by which to adjust the logits of specified tokens.
        :param shift_constant: Constant value by which to adjust the logits of specified tokens.
        :param selection_threshold: Threshold for selecting topic tokens.
        :param topical_encouragement: Factor by which to adjust the logits of selected topic tokens.
        :param args: Additional arguments passed to the base class generate method.
        :param kwargs: Additional keyword arguments passed to the base class generate method.
        :return: Tensor of generated token IDs.
        """
        experiment_name = EXPERIMENT_CONFIG["experiment_name"]

        if experiment_name == "factor_scaling":
            if scaling_factor is None:
                raise ValueError(
                    "factor_scaling parameter must be provided for factor scaling experiment."
                )
            topic_processor = FactorScalingLogitsProcessor(
                topic_token_ids=topic_token_ids, scaling_factor=scaling_factor
            )
        elif experiment_name == "constant_shift":
            if shift_constant is None:
                raise ValueError(
                    "constant_shift parameter must be provided for constant shift experiment."
                )
            topic_processor = ConstantShiftLogitsProcessor(
                topic_token_ids=topic_token_ids, shift_constant=shift_constant
            )
        elif experiment_name == "threshold_selection":
            if selection_threshold is None:
                raise ValueError(
                    "threshold_selection parameter must be provided for threshold selection experiment."
                )
            if topical_encouragement is None:
                raise ValueError(
                    "topical_encouragement parameter must be provided for threshold selection experiment."
                )
            topic_processor = ThresholdSelectionLogitsProcessor(
                topic_token_ids=topic_token_ids,
                selection_threshold=selection_threshold,
                topical_encouragement=topical_encouragement,
            )
        else:
            raise ValueError(f"Invalid experiment name: {experiment_name}")

        existing_processors = kwargs.get("logits_processor", [])
        kwargs["logits_processor"] = existing_processors + [topic_processor]

        return self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs
        )


def generate_summaries(
    dataloader: DataLoader,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    lda: Optional[LdaModel],
) -> List[Dict[str, Union[int, str]]]:
    """
    Generates abstractive summaries for a set of articles using a custom model that reweights token
    logits based on topic relevance.

    :param dataloader: The DataLoader containing the articles to be summarized.
    :param model: The model used to generate summaries.
    :param tokenizer: The tokenizer used to process text inputs.
    :param device: The device on which the model is running.
    :param lda: The LDA model used to determine topic focus, if any.
    :return: A list of dictionaries containing the generated summaries for each article.
    """
    try:
        experiment_name = EXPERIMENT_CONFIG["experiment_name"]
        model_alias = EXPERIMENT_CONFIG["model_alias"]

        logger.info(f"Started generating summaries using {model_alias} model.")

        experiment_information = {
            "EXPERIMENT_CONFIG": EXPERIMENT_CONFIG,
            "GENERATION_CONFIG": GENERATION_CONFIG,
            "DATASET_CONFIG": DATASET_CONFIG,
            "TOPICS_CONFIG": TOPICS_CONFIG,
        }

        if experiment_name == "factor_scaling":
            experiment_information["FACTOR_SCALING_DICT"] = FACTOR_SCALING_DICT
            factors = FACTOR_SCALING_DICT["scaling_factors"]
        elif experiment_name == "constant_shift":
            experiment_information["CONSTANT_SHIFT_DICT"] = CONSTANT_SHIFT_DICT
            factors = CONSTANT_SHIFT_DICT["shift_constants"]
        elif experiment_name == "threshold_selection":
            experiment_information[
                "THRESHOLD_SELECTION_DICT"
            ] = THRESHOLD_SELECTION_DICT
            factors = THRESHOLD_SELECTION_DICT["selection_thresholds"]
        else:
            raise ValueError(f"Invalid experiment name: {experiment_name}")
        topical_encouragement = THRESHOLD_SELECTION_DICT["topical_encouragement"]

        results = {"experiment_information": experiment_information}
        generated_summaries = []

        num_articles = DATASET_CONFIG["num_articles"]
        num_topic_words = TOPICS_CONFIG["num_topic_words"]
        batch_size = DATASET_CONFIG["batch_size"]

        model.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            for i, batch in enumerate(dataloader):
                if i * batch_size >= num_articles:
                    break

                article, article_idx = (
                    batch["article"][0],
                    batch["article_idx"][0].item(),
                )
                tid1 = batch["tid1"][0].item()

                article_dict = {
                    "article_idx": article_idx,
                    "tid1": tid1,
                }

                topic_tokens = get_topic_tokens(
                    tokenizer=tokenizer,
                    lda=lda,
                    tid=tid1,
                    num_topic_words=num_topic_words,
                )

                prompt = generate_prompt(article=article)
                tokenized_prompt = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=model.model.config.max_length,
                ).to(device)

                # Set parameters for generation as defined in the config.py file
                generation_config = get_generation_config(
                    model_alias=model_alias, tokenizer=tokenizer
                )

                for factor in factors:
                    output = model.generate(
                        input_ids=tokenized_prompt["input_ids"],
                        attention_mask=tokenized_prompt["attention_mask"],
                        topic_token_ids=topic_tokens,
                        scaling_factor=factor,
                        shift_constant=factor,
                        selection_threshold=factor,
                        topical_encouragement=topical_encouragement,
                        **generation_config.to_dict(),
                    )

                    # summary = tokenizer.decode(output[tokenized_prompt['input_ids'].shape[1]:], skip_special_tokens=False)
                    summary = tokenizer.decode(
                        output.squeeze(), skip_special_tokens=False
                    )
                    article_dict[str(factor)] = summary

                generated_summaries.append(article_dict)
        logger.info(
            f"Generated {len(factors)} summaries, for each of the {num_articles} articles."
        )
        results["generated_summaries"] = generated_summaries
        return results
    except Exception as e:
        logger.error(f"Error generating summaries: {e}")
        raise


def main():
    torch.manual_seed(0)
    experiment_name = EXPERIMENT_CONFIG["experiment_name"]
    model_alias = EXPERIMENT_CONFIG["model_alias"]

    if experiment_name not in [
        "factor_scaling",
        "constant_shift",
        "threshold_selection",
    ]:
        logger.error(f"Invalid experiment name: {experiment_name}")
        raise ValueError(f"Invalid experiment name: {experiment_name}")

    lda = load_lda()
    dataloader = setup_dataloader(dataset_name=DATASET_CONFIG["dataset_name"])
    tokenizer, model, device = load_model_and_tokenizer(CustomModel=CustomModel)

    summary_results = generate_summaries(
        dataloader=dataloader, model=model, tokenizer=tokenizer, device=device, lda=lda
    )

    # Save summaries to a file in the results_logits_reweighting directory
    file_name = (
        f"{experiment_name}_{model_alias}_{DATASET_CONFIG['num_articles']}_"
        f"{GENERATION_CONFIG['min_new_tokens']}_{GENERATION_CONFIG['max_new_tokens']}_"
        f"{GENERATION_CONFIG['num_beams']}.json"
    )

    results_dir = os.path.join(
        parent_dir, "data", f"results_{experiment_name}", file_name
    )
    with open(results_dir, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=4)
    logging.info("Summaries generated and stored.")


if __name__ == "__main__":
    main()
