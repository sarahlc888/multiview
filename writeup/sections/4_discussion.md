- [ ] Clear performance hierarchy: LM judge > embeddings > BM25

Using clever class naming and prompting, its possible to stretch “in-one-word” techniques quite far

Especially for "extraction-style" attributes where thinking is not required
(Aka the places it is easy to make data -- which could be a contributing factor to the popularity TBH)

But this method is very brittle

For example
Sentiment analysis
Positive or negative are clear labels

But for the HN task
A clear schema definition is an essential prereq









### Note:
The probe is a theoretical max on the landmark synthesis thing


### Discussion: Expressivity per criteria per embedding
Representation quality varies depending on the compute budget allocated to each data point / document.
The budget per document is typically 1 forward pass, both for Instructor-style models as well as to causal LM derived methods such as PromptEOL and [PonTE](https://arxiv.org/pdf/2504.16411).

A single forward pass is sufficient to capture simple criteria of the complexity used in CSTS for example (and even multihop compoutations in stronger models, cite Greenblatt)

However, complex criteria can require a larger compute budget per data point
e.g. BRIGHT -> query expansion is king

(Query rewriting has a one time ingestion cost)

For simple attributes, a linear projection is sufficient.
For complex attributes … we must actually finetune the embedder and/or put suey expansion in the loop

### Discussion: Why multiple views of the same data?
