# Task: Political Dogwhistles Dataset

## Goal
Create a dataset for identifying and categorizing coded political language (dogwhistles) in political texts. Enable multiview evaluation across different types of coded messaging, ideological signals, and rhetorical strategies.

## Type
ADD_DATASET

## Background
Political dogwhistles are coded phrases that communicate specific messages to target audiences while maintaining plausible deniability to others. The Allen AI Dogwhistles project (https://dogwhistles.allen.ai/) provides a glossary of documented dogwhistles with explanations.

## Approach
Use the Allen AI dogwhistles glossary as the dataset (concrete available data source):
- **Documents**: Individual dogwhistle terms/phrases from the glossary with their explanations
- **Criteria**:
  - `political_ideology` — which political ideology primarily uses this dogwhistle (conservative, progressive, libertarian, etc.)
  - `target_audience` — who is the intended in-group audience
  - `semantic_gap` — how different is the surface meaning from the coded meaning (literal vs. highly coded)
  - `rhetorical_function` — what the dogwhistle accomplishes (fear appeal, in-group signaling, othering, scapegoating, etc.)
  - `policy_domain` — what policy area it relates to (immigration, crime, economics, social issues, race, etc.)
  - `subtlety_level` — how coded/subtle vs. overt the language is
  - `historical_context` — what era or political moment this dogwhistle emerged from or is associated with

## Data source
- Allen AI Dogwhistles glossary: https://dogwhistles.allen.ai/glossary
  - Structured glossary with terms, definitions, and context
  - Can be scraped/parsed from the web interface or underlying data

## Implementation steps
- [ ] Scrape or fetch Allen AI dogwhistles glossary data
- [ ] Structure data format (term, definition, examples, context)
- [ ] Create new docset in `src/multiview/docsets/dogwhistles.py`
- [ ] Define criteria in `configs/available_criteria.yaml`
- [ ] Handle sensitive content considerations (political bias, potentially offensive language)
- [ ] Add data loading/caching logic
- [ ] Test eval creation with new docset

## Considerations
- **Bias and sensitivity**: This dataset inherently deals with political content and coded language that may be offensive or controversial
- **Neutrality**: Criteria descriptions should be analytically neutral while acknowledging the coded nature of the language
- **Scope**: May want to limit to specific time period or political context (e.g., US politics 2000-2024)
- **Updates**: Political language evolves quickly; glossary may need periodic updates

## Reference files
- `src/multiview/docsets/base.py` — BaseDocSet pattern
- `configs/available_criteria.yaml` — where criteria definitions live
- Existing docsets for text-based data (e.g., `moral_fables.py`, `tao_te_ching.py`)

## Exit criteria
- [ ] New docset implemented in `src/multiview/docsets/`
- [ ] Criteria defined in `configs/available_criteria.yaml`
- [ ] Can run `python scripts/create_eval.py` with dogwhistles docset
- [ ] Documentation of data source and any political content considerations
