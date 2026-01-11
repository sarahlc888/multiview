"""Criterion metadata for document sets.

This module contains metadata for LM-based criteria annotations across different
document sets. Each criterion includes:
- description: What the criterion measures
- pairwise_sim_hint: Hint for generating pairwise similarity comparisons (optional)
- category_schema_hint: Guidance for creating category-based schemas
- tag_schema_hint: Guidance for creating tag-based schemas
- summary_hint: Combined guidance + desired format for summaries (optional)
- triplet_example_hint: Example triplet guidance for LM triplet selection (optional)
"""

# GSM8K: Math word problems
GSM8K_CRITERIA = {
    "arithmetic": {
        "description": "The exact sequence of arithmetic operations (addition, subtraction, multiplication, division, etc.) required to solve the problem.",
        "pairwise_sim_hint": "Two problems are similar if they require the same sequence of arithmetic operations, even if the problem context or numbers differ. Focus on the operation sequence (e.g., multiply then add vs. divide then subtract) rather than the story context.",
        "category_schema_hint": None,
        "tag_schema_hint": "Consider tags like step1_add, step1_sub, step1_mul, step1_div, step2_add, etc.",
        "summary_hint": "List the exact sequence of arithmetic operations in order, formatted as e.g. ['multiplication', 'addition', 'division']",
        # Optional: default example triplet guidance for the LM triplet judge.
        # Can be overridden per-task via Task config / YAML.
        "triplet_example_hint": {
            "anchor": "Question: Bill walks 0.5 mile south, then 0.75 mile east, and finally 0.5 mile south. How many miles is he, in a direct line, from his starting point?\nAnswer: Bill walks 0.5 mile south and then another 0.5 mile south. The total distance south is 0.5 + 0.5 = <<0.5+0.5=1>>1 mile. He also walks 0.75 mile east.\nThe square of the southward distance is 1 × 1 = <<1*1=1>>1. The square of the eastward distance is 0.75 × 0.75 = <<0.75*0.75=0.5625>>0.5625. The sum of the squares is 1 + 0.5625 = <<1+0.5625=1.5625>>1.5625.\nThe square root of 1.5625 is <<sqrt(1.5625)=1.25>>1.25.\n#### 1.25",
            "pos": "Question: A subway route is planned to run 1.9 km north, then 2.5 km west, and finally 0.3 km south. How far is the subway from its original position?\nAnswer: The subway moves 1.9 km north and then 0.3 km south. The net northward distance is 1.9 − 0.3 = <<1.9-0.3=1.6>>1.6 km. It also moves 2.5 km west.\nThe square of the northward distance is 1.6 × 1.6 = <<1.6*1.6=2.56>>2.56. The square of the westward distance is 2.5 × 2.5 = <<2.5*2.5=6.25>>6.25. The sum of the squares is 2.56 + 6.25 = <<2.56+6.25=8.81>>8.81.\nThe square root of 8.81 is <<sqrt(8.81)=2.97>>2.97.\n#### 2.97",
            "neg": "Question: Bill walks to his school located 0.5 miles south from his house. If the walk took 12 minutes, how fast was Bill, in miles per hour?\nAnswer: Bill walks 0.5 miles in 12 minutes. There are 60 minutes in an hour. The time in hours is 12 ÷ 60 = <<12/60=0.2>>0.2 hours.\nSpeed is distance divided by time. Bill's speed is 0.5 ÷ 0.2 = <<0.5/0.2=2.5>>2.5 miles per hour.\n#### 2.5",
        },
    },
    "problem_type": {
        "description": "The domain or context of the word problem (money, time, measurement, rates, probability, geometry, etc.).",
        "pairwise_sim_hint": "Two problems are similar if they share the same domain or real-world context (e.g., both about money/shopping, both about time/scheduling). Consider the setting and concepts involved rather than the specific numbers or operations.",
        "category_schema_hint": "Consider categories like: money/finance, time/scheduling, distance/speed, measurement/units, ratios/proportions, geometry/area, probability, combinatorics, etc.",
        "tag_schema_hint": "Create tags for different problem domains that may overlap: involves_money, involves_time, involves_distance, involves_measurement, involves_rates, involves_geometry, involves_probability, etc.",
    },
    "solution_strategy": {
        "description": "The problem-solving approach or strategy needed (working backwards, setting up equations, using ratios, multi-step reasoning, etc.).",
        "category_schema_hint": "Consider categories like: direct computation, working backwards, setting up equations, using ratios/proportions, multi-step reasoning, logical deduction, etc.",
        "tag_schema_hint": "Create tags for different strategies: requires_working_backwards, requires_equation_setup, requires_ratio_reasoning, requires_multi_step, requires_unit_conversion, etc.",
    },
    "difficulty": {
        "description": "The relative difficulty or complexity of the problem based on steps required and concepts involved.",
        "category_schema_hint": "Consider categories like: simple (1-2 steps), moderate (3-4 steps), complex (5+ steps), or based on concept difficulty.",
        "tag_schema_hint": "Create tags for difficulty factors: multi_step, requires_intermediate_values, requires_complex_operations, requires_unit_conversion, has_multiple_entities, etc.",
    },
    "numerical_complexity": {
        "description": "The complexity of the numbers involved (whole numbers, decimals, fractions, large numbers, etc.).",
        "category_schema_hint": "Consider categories like: small whole numbers only, large whole numbers, decimals, fractions, mixed numbers, negative numbers, etc.",
        "tag_schema_hint": "Create tags for number types: has_whole_numbers, has_decimals, has_fractions, has_large_numbers, has_negative_numbers, has_small_numbers, etc.",
    },
}

# Crossword Clues
CROSSWORD_CRITERIA = {
    "wow_factor": {
        "description": "Crosswords are an indication of being tuned into arts and culture, general knowledge, and trivia. What 'bragging rights' would a person win for being able to solve this clue?",
    },
    "clue_type": {
        "description": "The type of crossword clue (independent of the topic). The archetype or prototype of the clue type. The kinds of strategies or 'moves' that clue writers might pull.",
        "category_schema_hint": "Consider categories like: straight definition, wordplay/pun, cryptic, fill-in-the-blank, trivia/knowledge, abbreviation, themed, and more.",
        # "Create tags for very specific clue properties: e.g. uses_wordplay, uses_abbreviation, requires_trivia, uses_definition, is_cryptic, is_themed, uses_anagram, is_playful_tone, is_serious_tone, is_simple, is_complicated, references_highly_obscure_fact, requires_domain_knowledge, etc.",
        "tag_schema_hint": "Create tags for a bunch of different types of clues: e.g. direct_definition (provides a standard definition or synonym for the word that is the answer), oblique_definition (provides a definition of the word that is the answer, but in a way that is very cryptic), direct_reference (refers directly to a notable person/event/object/place via their claim to fame), niche_reference (refers to a trivia factoid that would make the answer obvious, if known), anagram (uses an anagram of the answer), cross_reference (references to other parts of the puzzle), needs_letter (extremely hard and probably not able to be solved without knowing some of the letters), answer_is_word, answer_is_proper_noun, answer_is_abbrev, answer_is_phrase, answer_is_fragment, etc.",
    },
    "difficulty_factor": {
        "description": "What makes the clue difficult to solve? What is the source of uncertainty/ambiguity?",
        "category_schema_hint": "Consider categories like: niche_fact (Does the clue reference a specific piece of niche knowledge that would unlock the answer?), many_options (Does the clue narrow down to a particular class of items without providing a hint toward which one is the right answer?), unexpected_word_use (Is the clue misleading in that it requires the solver to interpret a word in an unconventional sense, which they would be unlikely to guess at first?), very_cryptic_needs_letters (Does the clue seem impossible to solve without knowing some of the letters?), puzzle_dependent (Is the clue difficult to solve because it is dependent on the puzzle as a whole, rather than just the clue itself?)",
        "tag_schema_hint": "Consider tags like: niche_fact (Does the clue reference a specific piece of niche knowledge that would unlock the answer?), many_options (Does the clue narrow down to a particular class of items without providing a hint toward which one is the right answer?), unexpected_word_use (Is the clue misleading in that it requires the solver to interpret a word in an unconventional sense, which they would be unlikely to guess at first?), very_cryptic_needs_letters (Does the clue seem impossible to solve without knowing some of the letters?), puzzle_dependent (Is the clue difficult to solve because it is dependent on the puzzle as a whole, rather than just the clue itself?). Also consider tags like is_a_thinker (you might be able to get it by thinking harder about it), know_it_or_dont (more time would not help solve); clear_clue (it's obvious what the clue says, you just don't know the answer), ambiguous_clue (it's not clear what the clue itself is even saying), misleading_clue (the clue is actively misleading in some way).",
    },
    # "description": "The method that the clue uses to hint at the answer. The way in which the answer is non obvious from just the clue, and what aspect of the necessary information is concealed.",
    "answer_domain": {
        "description": "The subject domain or topic of the answer (geography, history, pop culture, science, sports, etc.).",
        "category_schema_hint": "Consider categories like: geography, history, pop culture, science, sports, arts, politics, literature, general knowledge, etc.",
        "tag_schema_hint": "Create tags for different domains: geography, history, pop_culture, science, sports, arts, politics, literature, etc. Also consider properties like is_word, is_person, is_proper_noun.",
    },
    "difficulty": {
        "description": "The difficulty level of the clue based on obscurity and wordplay complexity.",
        "category_schema_hint": "Consider categories like: easy (common knowledge), medium (moderate knowledge/wordplay), hard (obscure or complex), etc.",
        "tag_schema_hint": "Create tags for difficulty factors: requires_specialized_knowledge, uses_complex_wordplay, uses_abbreviations, has_misdirection, etc.",
    },
    "answer_length": {
        "description": "The length category of the answer (short, medium, long).",
        "category_schema_hint": "Consider categories based on character count: very short (1-4 chars), short (5-7), medium (8-10), long (11+), etc.",
        "tag_schema_hint": "Create tags for length: single_word, multi_word, short_answer, long_answer, compound_word, etc.",
    },
}

# ROCStories: Short stories
ROCSTORIES_CRITERIA = {
    "narrative_arc": {
        "description": "The type of narrative structure or story arc (problem-solution, cause-effect, journey, conflict-resolution, etc.).",
        "category_schema_hint": "Consider categories like: problem-solution, cause-effect, journey/adventure, conflict-resolution, character growth, unexpected twist, etc.",
        "tag_schema_hint": "Create tags for narrative elements: has_problem, has_solution, has_conflict, has_resolution, has_twist, has_character_growth, etc.",
    },
    "theme": {
        "description": "The main theme or topic of the story (relationships, work, adventure, everyday life, challenges, etc.).",
        "category_schema_hint": "Consider categories like: relationships/family, work/career, adventure/travel, everyday life, overcoming challenges, humor, learning/growth, etc.",
        "tag_schema_hint": "Create tags for themes: involves_relationships, involves_work, involves_adventure, involves_everyday_life, involves_challenge, involves_humor, etc.",
    },
    "emotional_tone": {
        "description": "The emotional tone or mood of the story (positive, negative, neutral, humorous, serious, etc.).",
        "category_schema_hint": "Consider categories like: positive/uplifting, negative/sad, neutral/matter-of-fact, humorous, suspenseful, heartwarming, etc.",
        "tag_schema_hint": "Create tags for emotional qualities: is_positive, is_negative, is_humorous, is_serious, is_suspenseful, is_heartwarming, etc.",
    },
    "setting": {
        "description": "The setting or context where the story takes place (home, workplace, outdoor, travel, social, etc.).",
        "category_schema_hint": "Consider categories like: home/domestic, workplace/professional, outdoor/nature, travel, social/public, school/education, etc.",
        "tag_schema_hint": "Create tags for settings: at_home, at_work, outdoors, traveling, at_social_event, at_school, in_public, no_explicit_setting, implied_indoor_setting, vague_outdoor_setting, etc.",
    },
    "character_dynamics": {
        "description": "The types of character interactions and relationships in the story (solo, family, friends, strangers, etc.).",
        "category_schema_hint": "Consider categories like: solo protagonist, family interaction, friends, romantic, strangers/new relationships, professional colleagues, etc.",
        "tag_schema_hint": "Create tags for character elements: solo_protagonist, involves_family, involves_friends, involves_romance, involves_strangers, involves_colleagues, etc.",
    },
}

# Abstract-Sim: Wikipedia sentences with abstract descriptions
ABSTRACTSIM_CRITERIA = {
    "abstract_similarity": {
        "description": "The level of abstraction used to describe a concept. Sentences with similar abstraction levels use similar types of general or specific language, abstracting away from or focusing on particular details.",
        "pairwise_sim_hint": "Two sentences are similar if they describe concepts at the same level of abstraction, using similarly general or specific language. Focus on whether both texts abstract away from details to the same degree, not whether they share the same topic.",
        "category_schema_hint": "Consider categories based on abstraction level: highly specific (uses named entities, precise details, concrete examples), moderately specific (uses general categories with some detail), moderately abstract (uses broad concepts with minimal specifics), highly abstract (uses only general principles and abstract concepts).",
        "tag_schema_hint": "Create tags for abstraction properties: uses_named_entities, uses_specific_details, uses_concrete_examples, uses_general_categories, uses_broad_concepts, uses_abstract_concepts, describes_specific_instance, describes_general_pattern, includes_numbers_or_quantities, includes_dates_or_times, describes_action, describes_state, describes_relationship.",
    },
}
