- We could use this to generate triples to train on
- [ ] Productionized fine-tuning pipeline for triplet-based training.


- get a huge mix of training data and finetune an embedding model
	- no one is gonna use the model
- get a huge mix of training data and finetune a document rewriter
	- this is just gonna canonicalize the nomenclature


==we need to show a method that works well... otherwise the project is uninteresting?==
for anything classification adjacent, annotation -> vecEOL should kill
well we need a benchmark first, otherwise the project is nonexistent.......

Once we have this dataset, it useful not only for capability evaluations but also for training things...
- Training more capable embedding models (since it's a generic way of generating data for any unlabeled dataset)
- Training LMs to be better document rewriters (maybe with GEPA, so we don't have to actually finetune anything)
- Training zero shot probes or whatever for on the fly criteria-specific semantic similarity
- RL with triplet reward




hm but we want to be able to do this zero shot.....
maybe we should do a tuning method?
need some tuned methods
