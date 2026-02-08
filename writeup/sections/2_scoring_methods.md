# Methods for similarity comparison




TODO:
Just do a good job systematizing this per compute level...
(And tune where tuning can be allowed). In fact, add a table to summarize this


We consider a few different classes of methods.
1. Instruction-tuned embedding models
2. Instruction-tuned rerankers
3. "In-one-word" methods (PromptEOL, PonTE, etc)
4. Query rewriting + BM25
(Translate into something that embedding models will do a good job on )
5. LM judge (oracle / ceiling / sanity check)

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


## Baseline: instruction-tuned embedding models
- generation cost: O(1)
- embedding cost: O(n)

- no query rewriting

## Method: finetuned instruction-tuned embedding models
same as "Baseline: instruction-tuned embedding models", but with ranknet-style finetuning

## Baseline: instruction-tuned embedding models WITH query rewriting
- generation cost: O(n)
- embedding cost: O(n)

## Method: tuned query rewriter with frozen embedding model
- same as above, but tuned
- a solid local embedder model + gpt 5 mini query rewriter
- + tune the rewriter in various ways (e.g. GEPA)
- training a query rewriting "advisor model"
	- ==advisor models framing is cool? but query rewriting is nnot realistic in terms of cost? could be a good teacher though==
	- "Query rewriting is useful for more than just search" ... but the cost is intractible?
		- I'm honestly so confused about the query rewriting thing and why it's useful
		- It's not practical to do it for a huge corpus
		- Well unless it's like a 4B model in which case it is super practical to do so
		- Maybe we should distill query rewriting into the embedding model ?
		- Frame the query rewriting thing as a way to generate data. And focus on what cool examples this might help us unearth?


## Method: tuned query rewriter tuned frozen embedding model
- same as above, but tuned jointly


## Method: multiview embeddings

- method
	- For a newly provided criteria...
		- (1) Generate some new queries
		- (2) Select the most relevant queries from our query bank + new queries, using LLM judge scores
		- (3) Embed everything, and then produce our score embeddings as usual
- conceptual slop
	- Points of reference
	- Things relative to other things
	- Signposts
	- Landmark embeddings

==landmark = prototype = template== from https://cs231n.github.io/linear-classify/ "interpretation of ..."
## Method:

## Baselines
- (zero-shot) instruction-tuned embedding model
	- Re-embedding is the standard way to create multiple views of a dataset, and this is not done in practice

## New methods
- "advisor rewriters" (advisor models for query rewriting / expansion)
- landmark score vectors / similarity vectors
	- Instruction tuned embedding models -> limited sensitivity
	- Retrievers are better
	- What if we hijack the query expansion step? We can check where the embedding sits relative to a set of "landmarks/prototypes/queries"
