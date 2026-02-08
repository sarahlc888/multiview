# Task: Build graph structure from scientific abstract sentences

## Goal
Create graph representations from scientific abstract sentences, where edges represent co-occurrence or shared functional role.

## Type
OTHER

## Notes from original task
- "Let's make actual graphs out of these scientific abstracts"
- "prelabeled items should have ground truth labels, in the graph"
- Two types of relations:
  1. Do these sentences co-occur in the same abstract?
  2. Do these sentences play the same role within their abstract, as a whole?
- Also: "same thing for Tao Te Ching"

## Current state
- `src/multiview/docsets/arxiv_abstract_sentences.py` — ArxivAbstractSentencesDocSet already exists
- This docset likely already splits abstracts into sentences

## Reference files
- `src/multiview/docsets/arxiv_abstract_sentences.py` — existing sentence-level docset
- `src/multiview/docsets/arxiv_cs.py` — abstract-level docset
- This may need a new script rather than a new docset — graph construction is a different output format

## Steps
- [ ] Read `arxiv_abstract_sentences.py` to understand current sentence-level data
- [ ] Design graph structure: nodes = sentences, edges = co-occurrence or shared-role
- [ ] Decide format: networkx graph? adjacency list? JSON?
- [ ] Implement graph construction (may be a new script in `scripts/` rather than a docset)
- [ ] Add prelabeled ground truth where available
- [ ] Consider applying same approach to Tao Te Ching (cross-chapter sentence relationships)

## Exit criteria
- [ ] Graph can be constructed from abstract sentences
- [ ] Two edge types (co-occurrence, shared-role) are distinguishable
