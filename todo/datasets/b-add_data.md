# Add more datasets
- role-of-line-in-poem
Does similar work
What does it evoke
What hovers at the edges

- New Yorker Cartoons by why they're funny
- Quote co-occurence within essays but have to scrape
- museum exhibitions
- Paris Review interviews
- [ ] lines-in-poems dataset
- [ ] ==add the onion headlines==
- [ ] GSM8K math
- [ ] GSM8k set up
- [ ] Crossword topic
- [ ] Crossword clue type
- [ ] Haiku literal images
- [ ] Haiku evoked meaning
- [ ] Arxiv CS abstract science topic
- [ ] Arxiv CS abstract "approach"
    - a good science abstractions thing -- run the pipeline and see what kind of triplets we get
- "semantic filter" demo on top of HN
- Anthropic interview transcript demo focused on value alignment.
- https://dogwhistles.allen.ai/glossary#judicial_activism
- [ ] met
	- [ ] animacy
	- [ ] form
	- [ ] content


- [ ] put back the MET collection
	- [ ] artworks by common ancestors ("hidden influence network")
	- [ ] Paintings based on what is unusual about them
		- [ ] sparseness
	- [ ] Portraits based on expression
	- [ ] Portraits based on number of people in it
	- [ ] Portraits based on use of color



- [ ] semantic filter for HN (this should not be that hard)
	- hacker news by fun-factor
	- hacker news by personal blog
	- hacker news by angle of interest
	```
	https://news.ycombinator.com/

	Look at the top 100 stories per day
	Embed everything

	But based on what?
	If you just embed the headlines, you can probably get various groups

	People are kind of interested in AI stuff
	People are interested in space, medicine, etc

	But there's also a particular type of vibe WRT hacker news itself - like, why is it on hacker news

	And I feel like this will be cool and higher level "

	we need to be able to toggle between these views
	```


- [ ] Let's make actual graphs out of these scientific abstracts
	- [ ] prelabeled items should have ground truth labels, in the graph
		- Basically, we want two types of relations
			- Do these sentences co-occur in the same abstract?
			- Do these sentence play the same role within their abstract, as a whole?

- [ ] same thing for Tao Te Ching
	- https://github.com/nrrb/tao-te-ching/blob/master/Ursula%20K%20Le%20Guin.md

- [ ] "Deep semantic bitext mining"
	- [ ] ==let's do bitext mining between quotes from 2 authors==
		- [ ] schopenahur and nietzche
    this is harder because matches are sparser
- [ ] the d5 thing
- more better data
	- [ ] ==Add scientific abstract thing with an implicit taxonomy before doing vecEOL or rewrite then embed==
	- [ ] apply some taxonomy to some corpus - mmlu
	- [ ] similes by vibe

	- [ ] add that fashion mnist dataset or whatever it was... it should be really easy to see if the triplets are good!!
		- [ ] ==we do get embeddings as a byproduct of a bunch of these, so automatically tsne those up?==
	- [ ] `goodreads_quotes` - the positive sum thing might be a bit of a stretch.... don't do that one... instead do themes......
	- [ ] quickly run the aidanbench one
		- [ ] i feel like the responses are too toy though...? maybe filter for examples where the response is a bit longer??
		- [ ] it should provide a perfume taxonomy, why isn't it?
	- "NYCC ""moves"" - What is the angle?"
		- download data from https://nextml.github.io/caption-contest-data/ -> embed based on overt and covert features (a Dashboard is really necessary to show off the cool facter)
		- Can we add support for new data
- [ ] consider the edge case: very sparse matches with =="know it when you see it"==




- [ ] DEMO: get 1 thing of text mining
	- application: text mining
- [ ] DEMO: distribution over borges; over shakespeare
	- shakespeare-anything
	- borges-anything
	- a person has a finite set of points of reference
	- there is such a thing as a set of cultural touchstones http://www.incompleteideas.net/IncIdeas/BitterLesson.html

## Task categories by complexity

**Core capabilities to enable**
- Represent documents by arbitrary natural language instructions
- Pay ingestion cost once, query flexibly
- Embedding models with built-in reasoning capabilities
- Hypernetwork-style conditional embeddings

**Task types**
- Attribute extraction
- Abstractive reasoning / semantic group-by
- Deep semantic similarity beyond lexical/surface features

**Use case patterns**
- "Are these two responses equivalent with respect to X?"
- "Is description A applicable to instance B?"
- Finding unlikely semantic neighbors
- Identifying shared cultural/historical/thematic connections

### Extractive/categorical tasks
Simple reasoning over categorical criteria that can be extracted from documents.

**Reviews and sentiment**
- Movie reviews by sentiment
- Movie reviews by topic/semantic content
- Restaurant reviews by sentiment
- Restaurant reviews by aspect

**Intent classification**
- Banking queries by intent
- Chat prompts by degree of open-endedness

**Structured reasoning (math)**
- GSM8K problems by units of final answer
- GSM8K problems by arithmetic/mathematical structure of solution

**Code analysis**
- Code by big O complexity
- Code by big O complexity relative to optimal

### Multi-hop and compositional tasks
Tasks requiring multiple steps of reasoning or combining information.

**Crossword clues**
- By clue type
- By domain/subject matter of answer
- By answer itself
- By obscurity of answer
- By obfuscation strategy (regardless of topic)

**Creative content**
- Songs by key signature
- Songs by emotional tenor
- Poems by sentiment
- Poems by theme
- Technical blog posts by topic
- Technical blog posts by "voice-ey-ness"

**Academic content**
- Papers by methodology
- Abstracts by prototypical contribution

### Complex reasoning with sparse matches
Tasks where positive matches are rare or require deep semantic understanding.

**Narrative analysis**
- ROCStories by abstract narrative events
- ROCStories by explicit or implied setting
- Stories by source of tension
- Short stories by central theme
- Taxonomy of opening lines

**Visual art**
- Paintings by lighting quality
- Paintings by source of tension
- Paintings by implied narrative arc
- Paintings by "figures aware of being observed"

**Advanced semantic concepts**
- Garden path sentences by misdirection mechanism
- Subversion (weird in the same way)
- Thematic resonances
- Historical resonances
- Literary allusions
- Shared cultural references
- Anachronistic neighbors (e.g., "17th century painting close to 1980s photograph")

## Application domains

### Business applications

**Customer support**
- Chats by outcome
- Chats by topic
- Chats by implicit customer attitude toward AI chatbot

**Product and marketing**
- Product descriptions by market positioning
- Product descriptions by presumed market
- Recommendation systems ("You may also like")
- Business insights extraction

**Domain-specific use cases**
- Drug investments by expected return
- Drug investments by safety impacts
- Drug investments by customer satisfaction

### Research and technical applications

**Reasoning evaluation**
- PRM: reasoning traces by outcome
- Method embeddings: reasoning traces by process/method

**Existing research datasets**
- D5 news headlines
- AbstractSim pairs

### Visual/image classification

**Symbolic/archetypal concepts**
- Maslow's Hierarchy of Needs (5 levels)
- Big Five Personality Traits
- Seven Deadly Sins
- Ten Commandments
- Four Noble Truths (Buddhism)
- Four Classical Elements
- Seven Days of Creation
- Seven Basic Plots
- Five senses
- Nine muses
- Five stages of grief
- Seven circles of hell

**Temporal/aesthetic categorization**
- Decades aesthetics
- Anachronistic similarities


### Motivating examples and use cases

#### Real-world criteria-specific semantic similarity tasks
- Are these two responses equivalent with respect to XYZ?
- Is description ABC applicable to instance XYZ?
- "deep semantic similarity" - examples
- "types" of math problems, associative connections
- Unlikely Semantic neighbors
- Anachronistic neighbors: "this 17th century painting is weirdly close to this 1980s photograph"

#### Visual/image queries - fun queries to run through the image system
- paintings by TROPES
- Maslow's Hierarchy of Needs (5 levels)
- Big Five Personality Traits (5)
- Seven Deadly Sins (7)
- Ten Commandments (10)
- Four Noble Truths (4) (Buddhism)
- Four Classical Elements (4)
- Seven Days of Creation (7)
- Seven Basic Plots (7)
- The five senses
- The 9 muses
- Decades aesthetics
- In general, these clearly-clustered things kind of work well
	- The five stages of grief
	- The seven circles of hell

#### High-level motivations
- The ability to represent documents by arbitrary natural language instructions
- Pay an ingestion cost
- Embedding models should have reasoning -> generate embedding
- Hypernetwork

#### Specific domain applications
- PRM: reasoning traces by outcome
- lm_evals "method embeddings" - embed reasoning traces by method/process

### Business and application examples

#### Overall task types
- attribute extraction
- abstractive reasoning / semantic groupby
- "deep semantic similarity"

#### Customer support chats
- Customer support chats by outcome
- Customer support chats by topic
- Customer support chats by implicit customer attitude toward the AI chatbot

#### Product/business applications
- Product descriptions by market positioning
- Product descriptions by presumed market
- "You may also like" (recommendation systems)
- Business insights

#### Domain-specific applications
- Investments in drugs
	- By expected return
	- By safety impacts
	- By customer satisfaction

#### Research datasets
- abstractsim
- D5
