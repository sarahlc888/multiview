# Overflow: Fuzzy ideas from b-add_data.md

These items are notes/ideas that don't yet have enough definition for a standalone task. Revisit as inspiration strikes.

## Existing docsets that just need eval runs
- **Onion headlines**: docset + criteria already exist (`onion_headlines.py`, criteria: `topic`, `joke_type`). Just run eval.
- **GSM8K**: docset + criteria already exist (`gsm8k.py`, criteria: `final_expression`, `arithmetic`, `problem_type`, `complexity`). Just run eval.
- **"the d5 thing"** (line 79): D5 docset exists (`d5.py`). Unclear what additional work is needed.

## Vague ideas needing more definition
- **Quote co-occurrence within essays** (line 9): "but have to scrape" — need to identify the essay corpus
- **Museum exhibitions** (line 10): Separate from MET collection? Could be exhibition catalog data
- **Anthropic interview transcript demo** (line 24): "focused on value alignment" — need transcript source
- **Dogwhistles** (line 25): https://dogwhistles.allen.ai/glossary#judicial_activism — interesting reference, unclear what the task is
- **DEMO: text mining** (lines 99-100): "get 1 thing of text mining" — need to define what this means concretely
- **Lines-in-poems dataset** (line 12): Could be Dickinson-specific (covered in b01) or a broader poetry lines dataset
- **"know it when you see it" edge case** (line 94): Cross-cutting concern about sparse matches (relevant to bitext mining in b10)
- **"scientific abstract thing with implicit taxonomy"** (line 81): "Add scientific abstract thing with an implicit taxonomy before doing vecEOL or rewrite then embed" — partially covered by b04 (arxiv criteria) and b13 (abstract graphs). The vecEOL/rewrite-then-embed methodology is the key detail here.
- **"Can we add support for new data"** (line 93): Meta-question about making it easier to add new docsets



Rollouts by task-derived criteria
First paragraphs of op-eds by topic
First paragraphs of op-eds by "hook" strategy
ML abstract by topic/domain
ML abstract by "sensibility"


- poems
    ```
    KG anything


    Think of a big book like the Norton anthology of poetry. There are so many intertextual connections within those pages

    For example:
    Allusions from the title of one poem to famous lines from another poem
    Thematic resonances
    Poems that speak to each other despite being from very different times/places/authors
    Stylistic influences, shared sensibilities, subtle references

    And many other nuances

    I want to make an app that, given a corpus of text, creates a "map" or "knowledge graph" of the latent connections therein.

    Think about, like, OpenSyllabus.

    There are many ways to approach this, but one way is to create a criteria-conditional embedding model.



    ```



- [x] add anthropic data (available at `anthropiceconomicindex (skills)`)
    - download the data from [**https://www.anthropic.com/news/anthropic-economic-index-insights-from-claude-sonnet-3-7**](https://www.anthropic.com/news/anthropic-economic-index-insights-from-claude-sonnet-3-7)
    - [https://huggingface.co/datasets/Anthropic/EconomicIndex/viewer/default/train?p=29&row=2902](https://huggingface.co/datasets/Anthropic/EconomicIndex/viewer/default/train?p=29&row=2902)
- [ ] where is my cari data? i definitely scraped it at some point



media understanding	poems by theme
media understanding	Embed movies by vibe
media understanding	Embed movies by cinematography
media understanding	Embed movie by narrative style

multimodal (vision)	political cartoons


essence	embed story by summary ("Summarize based on main idea")


lm_evals	embed open-ended chat completions by {gpt-generated criteria}	aidanbench, chatbotarena, helm completions


instanceof	passage by rhetorical device	political speeches	y					types of fallacy; boilerplate sections of a speech; dogwhistles




referential / ancestral / intertextual	"stylistic influence/lineage/references" embeddings - Movies that reference the same movies; poems that reference the same poems; Comedy downstream of Late Night with Conan O'Brien					    - The most ambitious version of this would enable you to construct a genealogy of ideas / influence, like the pudding sampling data essay
referential / ancestral / intertextual	closest discernible stylistic influences					jack and the bean stalk; shakespeare retellings
task/strategy embeddings	EE 364 problems vs the algorithm needed to solve it
function embeddings							what is the function/utility of this snippet? e.g. exposition
theme / "intention/philosophy/didacticism" embeddings 	"Thematically but not explicitly connected to...."					the moralistic messages of movies






Task	Description	Data source	Labeled?	status	Priority (L, M, H)	Value add	plaintext instruction	Examples of clusters	related work
surprising connections / resemblance / affinity - ("recognition heuristic")	arxiv abstracts across time by surprising connections	arxiv abstracts	n			mining for nontrivial connections	What are some things that appear different at first glance but are actually fundamentally interconnected? We want non-obvious semantic connections that go beyond surface similarity
surprising connections / resemblance / affinity - ("recognition heuristic")	concepts by surprising connections	first paragraphs of wikipedia articles	n				"is there an interesting nonobvious connection, such that this might be an interesting idea to write an essay on"
taxonomy	pretraining documents by essential web tags	pretraining copora	y						ai2 weborganizer
"equivalence" embeddings	embed documents by NLI style entailment		n			semantic entropy
prototype / archetype / taxonomy / "moves"	Embed LSAT passage by archetype		n
prototype / archetype / taxonomy / "moves"	arxiv abstracts by implicit sensibility	arxiv						E.g. Scale is all you need
essence	"deep similarity" for arxiv abstracts ("sensibility"; "contribution")
toy	embed analogies by {}		n	tuned
toy	embed csts by {}		y	tuned
KG	embed conceptnet entities by relations (hypernym / hyponym / etc linguistics-flavor embeddings)	conceptnet	y
KG	abstracts by related work	citation graph	y
instanceof	embed by applicability	D5, abstractsim	y	ready to generate data	M
instanceof	Embed {passage, movie, entity} by trope	e.g. summaries of all simpson's episodes	y					Chekhov's Gun; The Hero's Journey
instanceof	movies by {narrative arc, theme, trope}	instanceof embeddings for tropes

instanceof	figurative language	should be plenty of stuff available - any classification dataset	y
"literary device" embeddings	stories by conceit
embed reasoning traces by method/process	https://www.anthropic.com/news/anthropic-economic-index-insights-from-claude-sonnet-3-7
"vibe" embeddings	aidanbench			ready to generate data			"do these things have the same vibe"
"vibe" embeddings	movies/aesthetic media			ready to generate data			"do these things have the same vibe"		Movies that make you feel a certain type of way
media understanding	documents (quotes) by authorship / persona						"Based on whether the same person would have said them"		https://arxiv.org/pdf/2411.18472
media understanding	embed jokes by {what makes them funny, how you could come up with them, subject matter} (zero shot humor classification)	onion headlines	dataset	ready to generate data
knowledge / named entities		aspectCSE paper
multimodal (vision)	text vs. picture of the text								mentioned here and in other modality gap work
referential / ancestral / intertextual 	narratives by shakespeare	rocstories lol?				retrieve shakespearian rocstories	"by relationship to x"
referential / ancestral / intertextual	allegory / plot structure embeddings; closest discernible narrative influences					jack and the bean stalk; shakespeare retellings



- swanson linking
	- PGD
	- NLP view of retrieval
	- Places where people reinvent the wheel

## in addition to rollouts, we should look at the following...
- "Archetype/move" embeddings
	- criteria: "the general type of contributions the paper makes"
	- texts: scientific abstractions
		- every academic discipline has “moves” - what type of paper per abstract? (hill climb, position, insights, etc)
- Vibe embeddings, vibe networks, vibe ancestors
	- criteria: `"general vibe of the ___"`
	- texts: (anything)
	- vibes
		- I want 2 things that are cool in the same way
		- if they appeal to you, it might appeal to you too
			- staircase paradox
			- langton's ant
- Entities
	- this is a bunch of stuff that's relatively easy to make data for?
	- https://web.stanford.edu/~jurafsky/slp3/6.pdf
- Moral embeddings
	- criteria: "overall moral, message, political sensibility, etc."
	- texts: movie titles, passages
	- intention
		- to get
		- Placing movies on the political compass
- boyd math https://github.com/cvxgrp/cvxbook_additional_exercises
- media by ...
	- politics
		- politics in children's media
			- https://booksforkeeps.co.uk/article/childrens-books-politics/
	- Network of influence within ART (but then have to just go on titles which is bad?)
	- people like to put stuff on the political compass, e.g. movies
		- https://www.reddit.com/r/PoliticalCompassMemes/comments/15bq5og/political_compass_of_movies_ive_seen_lately/
		- https://mhoweswrit340marketing.business.blog/
- could use weborganizer data too...  webtext by topic/format
- other...
	- intertextuality?
		- jstor or student essay quote co-occurence? where to find data like this
	- Easter eggs in movies - references, resonances, connections - combinatorial connections
	- art thieves
		- putting stuff die by side - thiebaud
			https://www.youtube.com/watch?v=p1enM09AuT8&ab_channel=FineArtsMuseumsofSanFrancisco
- text candidates
	- any criteria over the block quotes in wikipedia... those normally serve some cool purpose / are interesting in some way
	- cy twombly https://en.wikipedia.org/wiki/Cy_Twombly

- Short stories that address the same kind of epiphany
- Aesthetic experiences like the sigur ross thing and the clip from Zissou, except that's multimodal

- [ ] add more data sources
	- [ ] Find all of the old stuff that I have here collected in random files and shit
		- [ ] poke around in `/juice4/scr4/sachen/output_diversity/alpaca_eval/data/data/data.updated.json`
		- [ ] Write more of the semi-manual ones. Ask claude:
			Help me think of some things with unexpected but fundamental connections.
			Things that at first appear different
			But are fundamentally the same
			Surprising connections come up all the time, from literature to mathematics
	- not now
		- gsm8k: copy BARE setting exactly https://www.arxiv.org/pdf/2502.01697
		- or those papers that eval human creativity, so there is ground truth?
		- or something from fan wu paper (they look at goodreads and codecontests) https://arxiv.org/pdf/2407.02209
| Anchor         | Positive                     | Negative                            |
| -------------- | ---------------------------- | ----------------------------------- |
| (math problem) | (theorem needed to solve it) | (math probelm with the same set up) |


When you take on a particular point of view
Things group together in different interesting ways
Subtle associational vibes
By its sparkle
There are seven kinds of Tom and Jerry episodes
Take a book, chunk it up into paragraphs, and make a graph of connections and interpretations



Interesting features
```
* "Arguments that use appeals to authority vs. empirical evidence"
* "Explanations that build intuition vs. provide formal definitions"
* "Writing that assumes reader expertise vs. explains basics"
* "Responses that directly answer vs. reframe the question"
Invokes Foucault's concept of biopower
Poems that include a similar use of enjambment
```





Dominant color
Number of characters in the scene
Location of the vanishing point of the painting
Quadrant that has the most going on in it in terms of visual clutter
Religious themes
Subversion -- weird in the same way
Look alike -- plain surface similarity
Thematic
Historical
And then 2 that are generated on the fly based on what jumps out about the painting, e.g. narrative-ness
