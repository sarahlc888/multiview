"""Synthesis configurations for document sets.

This module contains synthesis prompts for LM-based document generation across
different document sets. Each criterion includes prompts for:
- hard_positive_prompt: Preserves criterion from Response 1, borrows themes from Response 2
- hard_negative_prompt: Changes criterion to Response 2, borrows themes from Response 1
"""

# GSM8K: Math word problems
GSM8K_SYNTHESIS_CONFIGS = {
    "arithmetic": {
        "hard_positive_prompt": """In this task, you will rewrite a math problem (Response 1) to preserve its arithmetic structure while incorporating thematic elements from a reference problem (Response 2).

## Example

**Input:**
Response 1:
Question: Cindy and Cheryl went to the candy store to buy treats for their second grade students. There are 23 kids in the class, and they bought a variety pack of 100 candies. If they give each kid 4 candies, how many candies will be left over?
Answer: They give out 23 * 4 = <<23*4=92>>92 candies.
There are 100 - 92 = <<100-92=8>>8 candies left over.
#### 8

Response 2:
Question: Rex the dog wanted to go to the park. Every day, when his owner Jack came home from work, there was a 20% chance that they went to the park together. What is the probability that they go to the park at least once this week?
Answer: The probability they don't go any day is 0.8^7 = <<0.8^7=0.2097>>0.2097.
The probability they go at least once is 1 - 0.2097 = <<1-0.2097=0.7903>>0.7903.
#### 0.7903

Criteria: arithmetic

**Expected Output:**
Question: Rex the dog loves to play at the park. Jack brought 100 tennis balls to the park. There are 23 other dogs at the park, and Rex shares 4 balls with each dog. How many tennis balls does Rex have left for himself?
Answer: Rex shares 23 * 4 = <<23*4=92>>92 tennis balls.
Rex has 100 - 92 = <<100-92=8>>8 tennis balls left.
#### 8

**Key points:**
- PRESERVE Response 1's arithmetic structure (subtraction and multiplication: 100 - 23*4)
- BORROW Response 2's themes (Rex the dog, Jack, park, tennis balls)

---

**Your Task:**

Response 1:
{text}

Response 2:
{ref}

Criteria: {criteria}

First, analyze the task:
1. What is Response 1's arithmetic structure that must be preserved?
2. What are Response 2's themes to incorporate?
3. How will you combine them?

Then, output your rewritten problem after the delimiter.

---FINAL OUTPUT---""",
        "hard_negative_prompt": """In this task, you will rewrite a math problem (Response 1) to use COMPLETELY DIFFERENT arithmetic operations while incorporating thematic elements from Response 2.

## Example

**Input:**
Response 1:
Question: Cindy and Cheryl went to the candy store to buy treats for their second grade students. There are 23 kids in the class, and they bought a variety pack of 100 candies. If they give each kid 4 candies, how many candies will be left over?
Answer: They give out 23 * 4 = <<23*4=92>>92 candies.
There are 100 - 92 = <<100-92=8>>8 candies left over.
#### 8

Response 2:
Question: Rex the dog wanted to go to the park. Every day, when his owner Jack came home from work, there was a 20% chance that they went to the park together. What is the probability that they go to the park at least once this week?
Answer: The probability they don't go any day is 0.8^7 = <<0.8^7=0.2097>>0.2097.
The probability they go at least once is 1 - 0.2097 = <<1-0.2097=0.7903>>0.7903.
#### 0.7903

Criteria: arithmetic

**Expected Output:**
Question: Cindy and Cheryl are planning a candy tasting activity. Each day for a week, there's a 75% chance that at least one student will try a new candy. What is the probability that students try candy at least once during the 7-day week?
Answer: The probability no students try candy on any day is 0.25^7 = <<0.25^7=0.00006>>0.00006.
The probability they try candy at least once is 1 - 0.00006 = <<1-0.00006=0.99994>>0.99994.
#### 0.99994

**Key points:**
- CHANGE to Response 2's arithmetic structure (exponentiation and subtraction: 0.8^7, 1 - result)
- BORROW Response 1's themes (Cindy, Cheryl, candy, students)

---

**Your Task:**

Response 1:
{text}

Response 2:
{ref}

Criteria: {criteria}

First, analyze the task:
1. What is Response 1's arithmetic structure (to avoid/change)?
2. What is Response 2's arithmetic structure (to adopt)?
3. What are Response 1's themes to incorporate?
4. How will you combine them?

Then, output your rewritten problem after the delimiter.

---FINAL OUTPUT---""",
    }
}
