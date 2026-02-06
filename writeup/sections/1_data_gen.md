
## Triplet format

- We evaluate using triplets to sidestep the challenge of globally calibrating pairwise similarity scores.
- Note: we create triplets that BM25 is unsuccessful on. This introduces bias, but __.
## Data generation


- A data generation pipeline that yields high quality triplets for arbitrary documents and criteria.

	- Input:
		- a set of unlabeled `documents`
		- a freetext `criteria`
	- Output:
		- triplets of the format `(anchor, positive, negative)`.

- The data-generation method is general-purpose and scaling-friendly. It is designed to generalize to arbitrary documents/criteria by using a stronger LM judge.
- For document sets with available annotation data, we also collect and repurpose data from pre-existing labeled datasets to create additional "gold" triplets.




## Data generation: synthesize -> propose -> annotate -> select -> validate

- Given `criteria: str`, `documents: List[str]`
	1. (optional) if true positive matches are expected to be sparse, **synthesize**documents to augment the dataset
	2. Use a strong LM to **propose** a "taxonomy"/"nomenclature"/"schema"
	3. **Annotate** each document with respect to how it relates to the criteria/schema
	4. **Select** true positives and negatives
	5. **Validate** triplets
### Synthesize
To augment a set of documents, we "remix" pairs of documents based on the criteria.
Currently, we do this only for GSM8K.

### Propose
Use a strong LM judge to reflect on how the criteria relates to the documents

We take inspiration from the popular approach of *proposing a taxonomy*.
Given a criteria and a set of unlabeled documents, we use an LM judge to propose a detailed taxonomy of how documents may relate to that criteria.
The taxonomy is privileged information that is used to create triplets but is not accessible to models at evaluation time.
### Annotate
Annotate each document with its relationship to the criteria using (a) a categorical assignment (b) boolean tags and (c) freetext summary
fuzzy summary
### Select
Randomly select triplet anchors and create triplets.
- For categorical annotations, create odd-one-out triplets.
- For tags, for each anchor, use jaccard similarity over tags to propose true positive candidates. Use an LM judge to select a true positive from the candidates.
- For freetext summaries, use an LM judge to select a hard negative from lexically similar candidates (based on BM25 scores)
### Validate
- Validate triplets with LM judge
- Discard low quality or borderline ambiguous triplets

Note: An over-reliance on LM judges is a the central weakness of this approach. However, even for a flawed LM judge, the performance gap between an LM judge and an embedding model is so large that this seems likely to still be a useful measure.

## rehash
- Details of annotation pipeline

	- Given the corpus and criteria, "reflect" on the criteria to generate (1) a more detailed description of what it means and how it applies to the dataset (2) some examples of document pairs that are similar with respect to the criteria and dissimilar with respect to the criteria (3) a category schema and (4) a tag schema
	- Then, annotate every single document in the corpus with how it relates to the criteria. We will use 3 kinds of annotations. (1) a one hot vector for categorical classification (2) a binary vector for tag applicability and (3) a free text summary of how the criteria relates to the document
	- We will then create triplets by randomly sampling an anchor item, retrieving the top-k most similar candidates based on jacquard similarity of the binary tag applicability vector, using an LM judge to select the best positives, and then also using an LM judge to select hard negatives from a pool of bm25-retrieved top-k candidates.


	- I've designed it to be very overpowered -- I want it to seem almost guaranteed to be effective.

## rehash

- rehash
	- Annotate every single item in the corpus with its relationship to the criteria
	- Use this as a very rough signal to surface
	- Use those annotations to find true positives to construct triplets with
	- To do hard negatives, we can use BM 25 on the raw documents



	- This should be more than powerful enough to take care of almost all the applications that we care about

	- There is an open question of how exactly we want to annotate. The most expressive way to do this is to make it a free text field that the model can fill out with anything.

	- A way that’s slightly more structured is to allow the model to create a set of tags, kind of like open set labeling with multi label, and then we’d likely be able to construct triplets ourselves without an LM call, which might be better, but honestly, whatever works is fine

	- The problem is that for a lot of complex criteria, it’s a bit difficult to reduce it to tags
	- A lot of stuff really isn’t categorical, but maybe it’s more prototypical than I think
