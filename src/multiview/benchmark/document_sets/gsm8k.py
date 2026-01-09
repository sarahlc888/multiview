"""GSM8K document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.benchmark.document_sets.base import BaseDocSet
from multiview.benchmark.document_sets.criteria_metadata import GSM8K_CRITERIA

logger = logging.getLogger(__name__)


class GSM8KDocSet(BaseDocSet):
    """GSM8K math word problems document_set."""

    # Metadata
    DATASET_PATH = "openai/gsm8k"
    DESCRIPTION = "GSM8K math word problems"

    # Criteria that can be extracted deterministically (no LLM needed)
    # word_count is automatically included by base class
    KNOWN_CRITERIA = []

    # Metadata for LM-based criteria (descriptions and schema hints)
    CRITERION_METADATA = GSM8K_CRITERIA
    # Synthesis prompts for LM-based document generation
    SYNTHESIS_CONFIGS = {
        "arithmetic": {
            "hard_positive_prompt": """In this task, you will rewrite a math problem (Response 1) to preserve its arithmetic structure while incorporating thematic elements from a reference problem (Response 2).

## Example

### Response 1
Question: Cindy and Cheryl went to the candy store to buy treats for their second grade students. There are 23 kids in the class, and they bought a variety pack of 100 candies. If they give each kid 4 candies, how many candies will be left over?
Answer: They give out 23 * 4 = <<23*4=92>>92 candies.
There are 100 - 92 = <<100-92=8>>8 candies left over.
#### 8

### Response 2
Question: Rex the dog wanted to go to the park. Every day, when his owner Jack came home from work, there was a 20% chance that they went to the park together. What is the probability that they go to the park at least once this week?
Answer: The probability they don't go any day is 0.8^7 = <<0.8^7=0.2097>>0.2097.
The probability they go at least once is 1 - 0.2097 = <<1-0.2097=0.7903>>0.7903.
#### 0.7903

### Criteria
arithmetic

### Rewrite Analysis
1. Response 1 arithmetic: subtraction and multiplication (100 - 23*4)
2. Response 2 themes: Rex the dog, Jack the owner, park, going to park
3. Rewrite to PRESERVE Response 1's arithmetic (subtraction and multiplication) but use Response 2's themes (Rex, Jack, park)

### Rewritten Problem
Question: Rex the dog loves to play at the park. Jack brought 100 tennis balls to the park. There are 23 other dogs at the park, and Rex shares 4 balls with each dog. How many tennis balls does Rex have left for himself?
Answer: Rex shares 23 * 4 = <<23*4=92>>92 tennis balls.
Rex has 100 - 92 = <<100-92=8>>8 tennis balls left.
#### 8

## Now rewrite these problems

### Response 1
{text}

### Response 2
{ref}

### Criteria
{criteria}

Output the rewritten problem directly, following the same format as the example.""",
            "hard_negative_prompt": """In this task, you will rewrite a math problem (Response 1) to use COMPLETELY DIFFERENT arithmetic operations while incorporating thematic elements from Response 2.

## Example

### Response 1
Question: Cindy and Cheryl went to the candy store to buy treats for their second grade students. There are 23 kids in the class, and they bought a variety pack of 100 candies. If they give each kid 4 candies, how many candies will be left over?
Answer: They give out 23 * 4 = <<23*4=92>>92 candies.
There are 100 - 92 = <<100-92=8>>8 candies left over.
#### 8

### Response 2
Question: Rex the dog wanted to go to the park. Every day, when his owner Jack came home from work, there was a 20% chance that they went to the park together. What is the probability that they go to the park at least once this week?
Answer: The probability they don't go any day is 0.8^7 = <<0.8^7=0.2097>>0.2097.
The probability they go at least once is 1 - 0.2097 = <<1-0.2097=0.7903>>0.7903.
#### 0.7903

### Criteria
arithmetic

### Rewrite Analysis
1. Response 1 arithmetic: subtraction and multiplication (100 - 23*4)
2. Response 2 arithmetic: exponentiation and subtraction (0.8^7, 1 - result)
3. Response 1 themes: Cindy, Cheryl, candy store, students
4. Rewrite to use Response 2's arithmetic (exponentiation, subtraction) but with Response 1's themes (Cindy, Cheryl, candy, students)

### Rewritten Problem
Question: Cindy and Cheryl are planning a candy tasting activity. Each day for a week, there's a 75% chance that at least one student will try a new candy. What is the probability that students try candy at least once during the 7-day week?
Answer: The probability no students try candy on any day is 0.25^7 = <<0.25^7=0.00006>>0.00006.
The probability they try candy at least once is 1 - 0.00006 = <<1-0.00006=0.99994>>0.99994.
#### 0.99994

## Now rewrite these problems

### Response 1
{text}

### Response 2
{ref}

### Criteria
{criteria}

Output the rewritten problem directly, following the same format as the example.""",
        }
    }

    def load_documents(self) -> list[Any]:
        """Load GSM8K problems as documents from Hugging Face.

        Loads the GSM8K dataset and formats each example as:
        "Question: {question}\nAnswer: {answer}"

        Returns:
            List of formatted documents (problems)
        """
        logger.info(f"Loading GSM8K from Hugging Face: {self.DATASET_PATH}")

        # Determine if we should use streaming mode
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        use_streaming = max_docs is not None and max_docs < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(
                self.DATASET_PATH, "main", split=split, streaming=True
            )
            # Shuffle and take the first max_docs
            dataset = dataset.shuffle(seed=42).take(max_docs)
        else:
            dataset = load_dataset(self.DATASET_PATH, "main", split=split)
            if max_docs is not None:
                # Shuffle and slice for non-streaming mode
                dataset = dataset.shuffle(seed=42)

        # Format documents
        documents = []
        for i, example in enumerate(dataset):
            # Format as "Question: ...\nAnswer: ..."
            formatted_doc = (
                f"Question: {example['question']}\nAnswer: {example['answer']}"
            )
            documents.append(formatted_doc)

            # Respect max_docs in non-streaming mode
            if not use_streaming and max_docs is not None and i + 1 >= max_docs:
                break

        logger.debug(f"Loaded {len(documents)} documents from GSM8K")
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A single document (formatted string)

        Returns:
            The text content of the document
        """
        # Documents are already formatted as strings
        return document if isinstance(document, str) else ""
