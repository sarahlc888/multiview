# Methods for similarity comparison


## Baseline: all pairs
- Even if we were to expend the pairwise compute, a problem with direct LM judge / extraction is global calibration. When there's no definition of exact equivalence, what constitutes "very similar" pair? That's part of what we are looking to distill -- consistency within the global space
- example
	- It's hard to map-reduce this stuff with LM judges or pointwise LM scorers bc global calibration is still a problem
	- How intense is this painting -> get an intensity axis -> .... A bit lackluster

- *hide*
	- [ ] If we have like 1 million corpus entries, we cannot grade them all... we need to do approximations... that's what the triples thing is for! and also the tuples thing! and alos the corpus thing but smlaler ig
		- We've got 1000 math problems. If we're gonna do all pairs, that's 1 million comparisons. We can do this with a very small model, or we can embed them all separately and take dot product.
		- We've got 1000 math problems, and we won a group them by topic, and Math structure
		- We've got 1000 language model responses, and we want to group them by functional equivalent
		- One way of doing this is training a dedicated classifier. We want to classify they can do anything.


## TODO:
Just do a good job systematizing this per compute level...
(And tune where tuning can be allowed). In fact, add a table to summarize this


We consider a few different classes of methods.
1. Instruction-tuned embedding models
2. Instruction-tuned rerankers
3. "In-one-word" methods (PromptEOL, PonTE, etc)
4. Query rewriting + BM25
(Translate into something that embedding models will do a good job on )
5. LM judge (oracle / ceiling / sanity check)


## Counterposition: we should not create separate embeddings per attribute
Ideally, we want to unified embedding that can have all of this information together in superposition
and then, like in mteb clf tasks, the metric is: How well could a linear probe read out this property? (indeed, What is the classification procedure for MTEB?)

Maybe a hypernetwork or linear projection at the end

TBH yeah i agree, we should do it this way, but I feel liek current models are not expressive enough to do this on non-toy tasks


==need Hypernetwork==


### Ensembling
There are a lot of methods that are good and bad at different tasks.
If you want to make data, you should just ensemble, with your source of truth being a strong LM judge -> curate triplets.
If you want, you can tune on this data (probably) to get better performance.
But in reality, instructed retrieval seems to be the only use case that really matters -- in which case it's impractical to do like 50% of the things we're looking at here because you can't re-embed documents. We *can* still use this triplet data to improve performance but in general, ... there's already a lot of work on instructed retrieval that's probably a better starting point...
