from transformers import AutoTokenizer, AutoModelForCausalLM
from prompter import Prompter

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MAX_LENGTH = 1024


class LlamaModelHandler:
    def __init__(
        self,
        init_examples_path: str,
        feedback_examples_path: str,
        iter_examples_path: str,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.prompter = Prompter(
            init_examples_path, feedback_examples_path, iter_examples_path
        )

    def generate_response(self, prompt: str, max_length: int = MAX_LENGTH) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output_ids = self.model.generate(
            input_ids, max_length=max_length, num_return_sequences=1
        )
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response

    def iterative_refinement(
        self, words: list[str], document: str, iterations: int = 3
    ) -> str:
        self.prompter.reset_iter()

        # Generate initial prompt and summary
        initial_prompt = self.prompter.create_init_prompt(words, document)
        summary = self.generate_response(initial_prompt)

        for _ in range(iterations - 1):
            # Request and process feedback for new summary
            feedback_prompt = self.prompter.create_feedback_prompt(
                words, document, summary
            )
            feedback = self.generate_response(feedback_prompt)
            concept_feedback, commonsense_feedback = self.prompter.extract_feedbacks(
                feedback
            )

            # Iterative refinement using previous summary and feedback
            iter_prompt = self.prompter.create_iter_prompt(
                words, document, summary, concept_feedback, commonsense_feedback
            )
            summary = self.generate_response(iter_prompt)

        return summary


def main():
    # Example usage with paths to your example files
    model_handler = LlamaModelHandler(
        init_examples_path="path/to/init_examples.json",
        feedback_examples_path="path/to/feedback_examples.json",
        iter_examples_path="path/to/iter_examples.json",
    )

    # Example inputs
    words = ["placeholder", "placeholder"]
    document = "placeholder"

    # Perform iterative refinement
    final_summary = model_handler.iterative_refinement(words, document, iterations=3)
    print("Final Summary:", final_summary)


if __name__ == "__main__":
    main()
