# Task: Add easy-to-verify image dataset

## Goal
Add a simple image classification dataset (Fashion-MNIST or similar) where triplet quality is easy to verify visually.

## Type
NEW_DOCSET

## Notes from original task
- "add that fashion mnist dataset or whatever it was... it should be really easy to see if the triplets are good!!"
- "we do get embeddings as a byproduct of a bunch of these, so automatically tsne those up?"
- Key value: QA/sanity check for the triplet generation pipeline on images

## Data source
- Fashion-MNIST (HuggingFace: `fashion_mnist`) — 10 clothing categories, 28x28 grayscale
- Or: CIFAR-10, or similar easily-classifiable image dataset
- UT-Zappos already exists (`ut_zappos50k.py`) — compare with that approach

## Reference files
- `src/multiview/docsets/ut_zappos50k.py` — existing image docset (shoes), good pattern to follow
- `src/multiview/docsets/met_museum.py` — another image docset with `get_document_image()`
- `src/multiview/docsets/base.py` — BaseDocSet pattern
- `src/multiview/docsets/__init__.py` — register

## Steps
- [ ] Create `src/multiview/docsets/fashion_mnist.py` with class `FashionMNISTDocSet`
  - Load from HuggingFace
  - Save images locally for `get_document_image()` to return paths
  - KNOWN_CRITERIA can include `category` (deterministic from labels)
- [ ] Register in `__init__.py`
- [ ] Add `fashion_mnist:` section to `configs/available_criteria.yaml`:
  - `visual_style`: Beyond category — what style attributes does this garment have?
- [ ] Run eval and visually inspect triplets for sanity

## Exit criteria
- [ ] Docset loads images and the pipeline generates triplets
- [ ] Triplets are visually verifiable (e.g., two t-shirts should be more similar than a t-shirt and a boot)
