# Pseudologits
Alternatively, do log prob rubric type thing ?


well, how do humans handle ambiguity?
proposing taxonomies
each criteria has an implicit taxonomy
and the taxonomy is the held out info that the evaluated model cannot see
but that the judge can see




Even though in-one-word style methods seem primitive, LMs are becoming increasingly powerful

Conditioned on a detailed taxonomy vs nomenclature (and/or demonstrations)
->
Can do pretty powerful things


so: Taxonomy thing is limited
and Class-based embeddings are extremely lossy
But ... probably it gets us really far





"Bucketization"

For conditional representations, people do this "in-one-word" hack

Which when you first look at it, just seems like... A terrible idea

But there's 2 things you can do to make this really powerful

First, come up with a really good taxonomy/schema and put it in context

Second, enable reasoning so that you can do non-trivial classification

What pops out of this is basically a logit vector of scores over classes, or buckets, or tags -- from whatever schema you defined

If you really care about getting criteria-conditional representations and have a lot of spare compute $, this approach can get you really far








The one thing that this doesn't fix is sparse matches
If you have really sparse matches
It's hard to write down a taxonomy that makes sense
You can try, but... some things really just seem to demand bi-encoder type stuff
To some extent, you SHOULD just use a reranker for this and want retrieval to be the "first step" filter only
