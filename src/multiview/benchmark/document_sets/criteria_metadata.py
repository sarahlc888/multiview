"""Criterion metadata for document sets.

This module contains metadata for LM-based criteria annotations across different
document sets. Each criterion includes:
- description: What the criterion measures
- category_schema_hint: Guidance for creating category-based schemas
- tag_schema_hint: Guidance for creating tag-based schemas
- summary_guidance_hint: Guidance for summarizing the criterion (optional)
- summary_format_hint: Format hints for summary output (optional)
"""

# GSM8K: Math word problems
GSM8K_CRITERIA = {
    "arithmetic": {
        "description": "The exact sequence of arithmetic operations (addition, subtraction, multiplication, division, etc.) required to solve the problem.",
        "category_schema_hint": None,
        "tag_schema_hint": None,
        "summary_guidance_hint": "List the exact sequence of arithmetic operations needed.",
        "summary_format_hint": "Return a list of operation types in order, e.g., ['multiplication', 'addition', 'division']",
    },
    "problem_type": {
        "description": "The domain or context of the word problem (money, time, measurement, rates, probability, geometry, etc.).",
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
    "clue_type": {
        "description": "The type or style of the crossword clue (definition, wordplay, cryptic, fill-in-the-blank, trivia, etc.).",
        "category_schema_hint": "Consider categories like: straight definition, wordplay/pun, cryptic, fill-in-the-blank, trivia/knowledge, abbreviation, themed, etc.",
        "tag_schema_hint": "Create tags for clue properties: uses_wordplay, uses_abbreviation, requires_trivia, uses_definition, is_cryptic, is_themed, etc.",
    },
    "domain": {
        "description": "The subject domain or topic of the clue (geography, history, pop culture, science, sports, etc.).",
        "category_schema_hint": "Consider categories like: geography, history, pop culture, science, sports, arts, politics, literature, general knowledge, etc.",
        "tag_schema_hint": "Create tags for different domains: geography, history, pop_culture, science, sports, arts, politics, literature, etc.",
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
        "tag_schema_hint": "Create tags for settings: at_home, at_work, outdoors, traveling, at_social_event, at_school, in_public, etc.",
    },
    "character_dynamics": {
        "description": "The types of character interactions and relationships in the story (solo, family, friends, strangers, etc.).",
        "category_schema_hint": "Consider categories like: solo protagonist, family interaction, friends, romantic, strangers/new relationships, professional colleagues, etc.",
        "tag_schema_hint": "Create tags for character elements: solo_protagonist, involves_family, involves_friends, involves_romance, involves_strangers, involves_colleagues, etc.",
    },
}
