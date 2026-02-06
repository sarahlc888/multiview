<img src="../../assets/glasses.jpg" width="200">

(rough: In general, it is a distinct task to identify semantic neighbors based on specific CRITERIA... + ideally doing so cheaply (without expensive re-embedding / re-indexing))


Unlike images, which can be compared at a glance, text-based similarity judgments often require explicit reasoning to extract properties of interest.

In theory, instruction-tuned embedding models can represent text with respect to arbitrary natural language instructions

In practice, embedding models prioritize general notions of semantic similarity and query-document relevance.



are not trivially extractible==. We
==so: new contribution...?==

- Current methods don't get at the core "value proposition" -- which is finding "surprising" semantic matches
        - "A different way of interacting with data..."


## Motivation

The motivation for a new "conditional similarity" benchmark is simple:
realistic documents are complex and multi-faceted.
Based on contextual factors, documents similarities should look very different.
Conditional embeddings is one way to approach this.

## aplication

https://www.lesswrong.com/posts/q9g9zuudd3Pvw2cbj/global-cot-analysis-initial-attempts-to-uncover-patterns-1

Clustering by sentence embedding


# trash can
In practice, many SOTA embedding models do not accept instructions, or accept instructions on the query side only



, e.g. two documents that are not superficially similar but have deeper semantic parallels.

 We hope that progress in this domain may spark more effective methods for
semantic bitext mining,
semantic deduplication,
and surfacing surprising semantic neighbors.

## Overview
- Standard embedding models are trained to prioritize query-document relevance and general semantic similarity
- Similarity and relevance judgments are highly **context-dependent**, but instruction following is typically deferred to the reranking stage (since it doesn't make sense to constantly re-embed docuemnts based on different criteria)
- This benchmark evaluates performance on **criteria-specific semantic similarity**

## trash
-
-  (e.g. [TODO]).
- Conditional similarity is established in the image domain,

	- Is it a finite, closed set of labels. If it’s not finite, is it at least a numerable?
	- Extractable.....


    GRACE
    https://github.com/GasolSun36/GRACE?tab=readme-ov-file#-data-processing-pipeline


- Triplets - (While annotating exact equivalence between pairs of items is feasible, it becomes more difficult to calibrate scores when similarity criteria is fuzzy.)

## viz



We also use
https://en.wikipedia.org/wiki/Self-organizing_map
inspired by
https://nry.me/posts/2025-10-09/small-web-screenshots/
