"""
logits_reweighting.py

This module contains a custom model class and multiple custom logits processors that are used 
to reweight logits based on their topic relevance.
"""

# Standard library imports
import logging
from typing import List, Optional

# Third-party imports
import torch
from transformers import LogitsProcessor, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
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
    def __init__(self, topic_token_ids, selection_threshold: float, topical_encouragement: float):
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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
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
        topic_probs = torch.where(topic_mask, probs, torch.tensor(0.0, dtype=dtype, device=device))

        # Identify topic tokens with probabilities above the threshold
        exceed_threshold_mask = topic_probs >= self.selection_threshold

        # Find the maximum logit value in each item of the batch
        max_logit, _ = torch.max(scores, dim=-1, keepdim=True)

        # Update logits for topic tokens largen than threshold to max_logit + topical_encouragement
        # We only update logits where exceed_threshold_mask is True
        scores = torch.where(exceed_threshold_mask, max_logit + self.topical_encouragement, scores)

        return scores

class CustomModel:
    """
    A model that uses a custom logits processor to reweight logits based on their topic relevance. 
    """
    def __init__(self, pretrained_model_name_or_path, *args, **kwargs):
        # Load the underlying model instance
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def generate(self, experiment_name: str, input_ids: List[int], attention_mask=None,
                 topic_token_ids=None, scaling_factor: Optional[float] = None,
                 shift_constant: Optional[float] = None,
                 selection_threshold: Optional[float] = None,
                 topical_encouragement: Optional[float] = None, *args, **kwargs):

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

        if experiment_name == 'factor_scaling':
            if scaling_factor is None:
                raise ValueError("factor_scaling parameter must be provided for factor scaling experiment.")
            topic_processor = FactorScalingLogitsProcessor(topic_token_ids=topic_token_ids,
                                                           scaling_factor=scaling_factor)
        elif experiment_name == 'constant_shift':
            if shift_constant is None:
                raise ValueError("constant_shift parameter must be provided for constant shift experiment.")
            topic_processor = ConstantShiftLogitsProcessor(topic_token_ids=topic_token_ids,
                                                           shift_constant=shift_constant)
        elif experiment_name == 'threshold_selection':
            if selection_threshold is None:
                raise ValueError("threshold_selection parameter must be provided for threshold selection experiment.")
            if topical_encouragement is None:
                raise ValueError("topical_encouragement parameter must be provided for threshold selection experiment.")
            topic_processor = ThresholdSelectionLogitsProcessor(topic_token_ids=topic_token_ids,
                                                                selection_threshold=selection_threshold,
                                                                topical_encouragement=topical_encouragement)
        else:
            raise ValueError(f"Invalid experiment name: {experiment_name}")

        existing_processors = kwargs.get("logits_processor", [])
        kwargs["logits_processor"] = existing_processors + [topic_processor]

        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
