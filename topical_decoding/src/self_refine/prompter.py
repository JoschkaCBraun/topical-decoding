import json


class Prompter:
    def __init__(
        self,
        init_examples_path: str,
        feedback_examples_path: str,
        iter_examples_path: str,
    ) -> None:
        self.separator = "\n\n\n###\n\n\n"

        # Init
        self.init_current_stripped_template = """Topic words: {words}

Document: {document}"""

        self.init_example_template = (
            self.init_current_stripped_template + "\n\nSummary: {summary}"
        )

        self.init_current_template = (
            self.init_current_stripped_template + "\n\nSummary: "
        )

        self.init_prefix = (
            "We want to create a summary that incorporates the specified words. "
            "Do your best! Put a large emphasis on aligning your "
            "summary with the topic words but also make sure you "
            "capture the essence of the document."
        )

        self.init_examples = self.create_init_examples(init_examples_path)

        # Feedback
        self.feedback_prefix = (
            "We want to create a summary that incorporates the specified words. "
            "Please provide feedback on the following sentences. "
            "You should give a topic alignment feedback and an article coverage "
            "feedback."
        )

        self.feedback_current_stripped_template = """Summary: {summary}

What words from the word list are missing from the summary? Are there non-stopwords in
the summary that are not in the word list? Does the summary make sense? Make sure to
follow the example templates exactly."""

        self.feedback_current_template = (
            "Topic words: {words}\n\nDocument: {document}\n\n"
            + self.feedback_current_stripped_template
            + "\n\nConcept Feedback: "
        )

        self.feedback_example_template = (
            self.feedback_current_template
            + "{concept_feedback}\n\nCommonsense Feedback: {commonsense_feedback}"
        )

        self.feedback_examples = self.create_feedback_examples(feedback_examples_path)

        # Iter
        self.iter_separator = (
            "\n\nOkay, now improve the summary using the feedback. Do your best!\n\n"
        )

        self.iter_history = ""

        self.iter_init_template = self.feedback_example_template + self.iter_separator

        self.iter_update_template = (
            f"{self.feedback_current_stripped_template}"
            "\n\nConcept Feedback: {concept_feedback}"
            "\n\nCommonsense Feedback: {commonsense_feedback}"
        )

        self.iter_examples = self.create_iter_examples(iter_examples_path)

    def create_init_examples(self, init_examples_path: str) -> str:
        with open(init_examples_path) as f:
            examples = json.load(f)["examples"]

        example_strings = [
            self.init_example_template.format(
                words=example["words"],
                document=example["document"],
                summary=example["summary"],
            )
            for example in examples
        ]

        return f"{self.separator.join(example_strings)}{self.separator}"

    def create_init_prompt(self, words: list[str], document: str) -> str:
        current_string = self.init_current_template.format(
            words=words, document=document
        )
        init_prompt = f"{self.init_prefix}\n\n\n{self.init_examples}{current_string}"

        return init_prompt

    def create_feedback_examples(self, feedback_examples_path: str) -> str:
        with open(feedback_examples_path) as f:
            examples = json.load(f)["examples"]

        example_strings = [
            self.feedback_example_template.format(
                words=example["words"],
                document=example["document"],
                summary=example["summary"],
                concept_feedback=example["concept_feedback"],
                commonsense_feedback=example["commonsense_feedback"],
            )
            for example in examples
        ]

        return f"{self.separator.join(example_strings)}{self.separator}"

    def create_feedback_prompt(
        self, words: list[str], document: str, summary: str
    ) -> str:
        current_string = self.feedback_current_template.format(
            words=words, document=document, summary=summary
        )
        feedback_prompt = (
            f"{self.feedback_prefix}{self.feedback_examples}{current_string}"
        )

        return feedback_prompt

    @staticmethod
    def extract_feedbacks(response: str) -> tuple[str, str]:
        concept_feedback, commonsense_feedback = response.split(
            "Commonsense Feedback: "
        )
        concept_feedback = concept_feedback.strip()
        commonsense_feedback = commonsense_feedback.strip()

        return concept_feedback, commonsense_feedback

    def reset_iter(self) -> None:
        self.iter_history = ""

    def create_iter_examples(self, iter_examples_path: str) -> str:
        with open(iter_examples_path) as f:
            examples = json.load(f)["examples"]

        sequences = []

        for example in examples:
            example_strings = [
                self.feedback_example_template.format(
                    words=sub_example["words"],
                    document=sub_example["document"],
                    summary=sub_example["summary"],
                    concept_feedback=sub_example["concept_feedback"],
                    commonsense_feedback=sub_example["commonsense_feedback"],
                )
                for sub_example in example
            ]
            sequences.append(self.iter_separator.join(example_strings))

        return (
            f"{self.init_prefix}\n\n\n{self.separator.join(sequences)}{self.separator}"
        )

    def create_iter_prompt(
        self,
        words: list[str],
        document: str,
        summary: str,
        concept_feedback: str,
        commonsense_feedback: str,
    ):
        if not self.iter_history:
            current_string = self.iter_init_template.format(
                words=words,
                document=document,
                summary=summary,
                concept_feedback=concept_feedback,
                commonsense_feedback=commonsense_feedback,
            )
        else:
            current_string = self.iter_update_template.format(
                summary=summary,
                concept_feedback=concept_feedback,
                commonsense_feedback=commonsense_feedback,
            )
        current_example = self.iter_history + "\n\n\n" + current_string

        self.iter_history = current_example

        iter_prompt = f"{self.iter_examples}{current_example}"

        return iter_prompt
