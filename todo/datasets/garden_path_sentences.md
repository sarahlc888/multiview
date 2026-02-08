# Garden Path Sentences

## Background

A garden-path sentence is a grammatically correct sentence that starts in such a way that the reader's most likely interpretation will be incorrect. The reader is lured into a parse that turns out to be a dead end, forcing re-analysis. Named after the idiom "to be led down the garden path" (= to be deceived).

Garden-path sentences are useful for evaluating how models handle syntactic ambiguity, incremental parsing, and reanalysis — relevant to multiview embedding because different parsing strategies should yield measurably different representations.

## Data Sources

1. **GP-mechanisms dataset** (Hanna et al.): `ambiguous=True` rows from [garden_path_readingcomp.csv](https://github.com/hannamw/GP-mechanisms/blob/main/data_csv/garden_path_readingcomp.csv)
2. **Wikipedia**: [Garden-path sentence](https://en.wikipedia.org/wiki/Garden-path_sentence)
3. **DJ suggestion**: Groucho Marx quote

---

## Classic Examples (Wikipedia)

| Sentence | Explanation |
|----------|-------------|
| The old man the boat. | "old" = noun (elderly people), "man" = verb (to operate). Rephrased: "Those who man the boat are old." |
| The complex houses married and single soldiers and their families. | "complex" = noun (housing complex), "houses" = verb (provides housing). |
| The horse raced past the barn fell. | "raced" = passive participle (reduced relative clause), not simple past. Rephrased: "The horse that was raced past the barn fell." |
| Time flies like an arrow; fruit flies like a banana. | "flies" shifts from verb to noun, "like" from preposition to verb. |

## From DJ

> One morning I shot an elephant in my pajamas. How he got into my pajamas I don't know.
> — Groucho Marx, *Animal Crackers*, 1930

PP-attachment ambiguity: "in my pajamas" can modify the subject ("I, wearing pajamas") or the object ("elephant wearing pajamas").

---

## GP-mechanisms Dataset (ambiguous=True rows)

From Hanna et al. — 72 ambiguous garden-path sentences with reading comprehension questions. Three condition types:

### Condition Types

| Condition | Name | Ambiguity Type | Example Pattern |
|-----------|------|---------------|-----------------|
| **NPS_UAMB** | NP/S | Sentential complement ambiguity | "X showed the file deserved..." — did X show the file, or did the file deserve...? |
| **NPZ_UAMB** | NP/Z | Clause boundary (subject/object) | "Because X changed the file deserved..." — did X change the file, or did the file deserve...? |
| **MVRR_UAMB** | Main Verb / Reduced Relative | MV vs. past-participle ambiguity | "X sent the file deserved..." — did X send the file, or was X who was sent the file...? |

Each item has a `Comp_Question_No` (answer is No) and `Comp_Question_Yes` (answer is Yes) to test comprehension.

### NPS_UAMB (24 sentences)

| # | Sentence | Q (answer=No) | Q (answer=Yes) |
|---|----------|---------------|----------------|
| 1 | The suspect showed the file deserved further investigation during the murder trial. | Did the suspect show the file? | Did the file deserve further investigation during the murder trial? |
| 2 | The corrupt politician mentioned the bill received unwelcome attention from southern voters. | Did the corrupt politician mention the bill? | Did the bill receive unwelcome attention from southern voters? |
| 3 | The woman maintained the mail disappeared mysteriously from her front porch. | Did the woman maintain the mail? | Did the mail disappear mysteriously from her front porch? |
| 4 | The boy found the chicken stayed surprisingly happy in the new barn. | Did the boy find the chicken? | Did the chicken stay surprisingly happy in the new barn? |
| 5 | The new doctor demonstrated the operation appeared increasingly likely to succeed. | Did the new doctor demonstrate the operation? | Did the operation appear increasingly likely to succeed? |
| 6 | The professor noticed the grant gained more attention from marine biologists. | Did the professor notice the grant? | Did the grant gain more attention from marine biologists? |
| 7 | The technician reported the service stopped working almost immediately after the storm started. | Did the technician report the service? | Did the service stop working almost immediately after the storm started? |
| 8 | The mechanic observed the truck needed several more hours to be repaired. | Did the mechanic observe the truck? | Did the truck need several more hours to be repaired? |
| 9 | The guitarist knew the song failed dramatically because of the tensions within the band. | Did the guitarist know the song? | Did the song fail dramatically because of the tensions within the band? |
| 10 | The player revealed the bonus remained essentially the same as in the original contract. | Did the player reveal the bonus? | Did the bonus remain essentially the same as in the original contract? |
| 11 | The recent hire claimed the job prepared many students for careers in media. | Did the recent hire claim the job? | Did the job prepare many students for careers in media? |
| 12 | The assistant manager discovered the training seemed unnecessarily demanding for new staff. | Did the assistant manager discover the training? | Did the training seem unnecessarily demanding for new staff? |
| 13 | The mayor showed the document provided sufficient evidence to prove her innocence. | Did the mayor show the document? | Did the document provide sufficient evidence to prove her innocence? |
| 14 | The basketball player mentioned the contract created another controversy in the NBA. | Did the basketball player mention the contract? | Did the contract create another controversy in the NBA? |
| 15 | The engineer maintained the equipment required constant supervision from senior technicians. | Did the engineer maintain the equipment? | Did the equipment require constant supervision from senior technicians? |
| 16 | The little girl found the lamb remained relatively calm despite the absence of its mother. | Did the little girl find the lamb? | Did the lamb remain relatively calm despite the absence of its mother? |
| 17 | The yoga instructor demonstrated the position demanded immense physical effort from everyone. | Did the yoga instructor demonstrate the position? | Did the position demand immense physical effort from everyone? |
| 18 | The governor noticed the contract received sweeping support across the entire state. | Did the governor notice the contract? | Did the contract receive sweeping support across the entire state? |
| 19 | The patient reported the treatment continued causing uncomfortable side effects like nausea. | Did the patient report the treatment? | Did the treatment continue causing uncomfortable side effects like nausea? |
| 20 | The operator observed the machine started working efficiently all of a sudden. | Did the operator observe the machine? | Did the machine start working efficiently all of a sudden? |
| 21 | The dancer knew the ballet achieved incredible success for a small local production. | Did the dancer know the ballet? | Did the ballet achieve incredible success for a small local production? |
| 22 | The contestant revealed the money became unavailable to him when the show's budget shrank. | Did the contestant reveal the money? | Did the money become unavailable to him when the show's budget shrank? |
| 23 | The new chef claimed the restaurant separated mediocre cooks from gifted ones. | Did the new chef claim the restaurant? | Did the restaurant separate mediocre cooks from gifted ones? |
| 24 | The apprentice baker discovered the oven produced smaller cakes because it heated too fast. | Did the apprentice baker discover the oven? | Did the oven produce smaller cakes because it heated too fast? |

### NPZ_UAMB (24 sentences)

| # | Sentence | Q (answer=No) | Q (answer=Yes) |
|---|----------|---------------|----------------|
| 1 | Because the suspect changed the file deserved further investigation during the jury discussions. | Did the suspect change the file? | Did the file deserve further investigation during the jury discussions? |
| 2 | After the corrupt politician signed the bill received unwelcome attention from southern voters. | Did the corrupt politician sign the bill? | Did the bill receive unwelcome attention from southern voters? |
| 3 | After the woman moved the mail disappeared mysteriously from the delivery system. | Did the woman move the mail? | Did the mail disappear mysteriously from the delivery system? |
| 4 | Although the boy attacked the chicken stayed surprisingly happy as if nothing happened. | Did the boy attack the chicken? | Did the chicken stay surprisingly happy as if nothing happened? |
| 5 | After the new doctor left the operation appeared increasingly likely to succeed. | Did the new doctor leave the operation? | Did the operation appear increasingly likely to succeed? |
| 6 | After the professor read the grant gained more attention due to her excellent description. | Did the professor read the grant? | Did the grant gain more attention due to her excellent description? |
| 7 | After the technician called the service stopped working almost immediately to his surprise. | Did the technician call the service? | Did the service stop working almost immediately to his surprise? |
| 8 | Because the mechanic stopped the truck needed several more hours before it could be fully repaired. | Did the mechanic stop the truck? | Did the truck need several more hours before it could be fully repaired? |
| 9 | After the guitarist began the song failed dramatically because he skipped the sound check. | Did the guitarist begin the song? | Did the song fail dramatically because he skipped the sound check? |
| 10 | Although the player lost the bonus remained essentially the same as in the original contract. | Did the player lose the bonus? | Did the bonus remain essentially the same as in the original contract? |
| 11 | Once the recent hire started the job prepared many students for careers in media. | Did the recent hire start the job? | Did the job prepare many students for careers in media? |
| 12 | While the assistant manager worked the training seemed unnecessarily demanding to him. | Did the assistant manager work the training? | Did the training seem unnecessarily demanding to him? |
| 13 | Although the mayor changed the document provided sufficient evidence for what he had promised. | Did the mayor change the document? | Did the document provide sufficient evidence for what he had promised? |
| 14 | After the basketball player signed the contract created another controversy in the NBA. | Did the basketball player sign the contract? | Did the contract create another controversy in the NBA? |
| 15 | After the engineer moved the equipment required constant supervision from senior technicians. | Did the engineer move the equipment? | Did the equipment require constant supervision from senior technicians? |
| 16 | When the little girl attacked the lamb remained relatively calm despite the sudden assault. | Did the little girl attack the lamb? | Did the lamb remain relatively calm despite the sudden assault? |
| 17 | Before the yoga instructor left the position demanded immense physical effort from everyone. | Did the yoga instructor leave the position? | Did the position demand immense physical effort from everyone? |
| 18 | While the governor read the contract received sweeping support from the audience at the rally. | Did the governor read the contract? | Did the contract receive sweeping support from the audience at the rally? |
| 19 | Before the patient called the treatment continued causing uncomfortable side effects like nausea. | Did the patient call the treatment? | Did the treatment continue causing uncomfortable side effects like nausea? |
| 20 | Once the operator stopped the machine started working efficiently without any supervision. | Did the operator stop the machine? | Did the machine start working efficiently without any supervision? |
| 21 | Once the dancer began the ballet achieved incredible success for a show with a new performer. | Did the dancer begin the ballet? | Did the ballet achieve incredible success for a show with a new performer? |
| 22 | After the contestant lost the money became unavailable despite his previous three wins in a row. | Did the contestant lose the money? | Did the money become unavailable despite his previous three wins in a row? |
| 23 | Once the new chef started the restaurant separated mediocre cooks from gifted ones. | Did the new chef start the restaurant? | Did the restaurant separate mediocre cooks from gifted ones? |
| 24 | When the apprentice baker worked the oven produced smaller cakes because he lacked experience. | Did the apprentice baker work the oven? | Did the oven produce smaller cakes because he lacked experience? |

### MVRR_UAMB (24 sentences)

| # | Sentence | Q (answer=No) | Q (answer=Yes) |
|---|----------|---------------|----------------|
| 1 | The suspect sent the file deserved further investigation given the new evidence. | Did the suspect send the file? | Did the suspect deserve further investigation given the new evidence? |
| 2 | The corrupt politician handed the bill received unwelcome attention from southern voters. | Did the corrupt politician hand the bill? | Did the corrupt politician receive unwelcome attention from southern voters? |
| 3 | The woman brought the mail disappeared mysteriously after reading the bad news in it. | Did the woman bring the mail? | Did the woman disappear mysteriously after reading the bad news in it? |
| 4 | The boy fed the chicken stayed surprisingly happy despite having a mild allergic reaction. | Did the boy feed the chicken? | Did the boy stay surprisingly happy despite having a mild allergic reaction? |
| 5 | The new doctor offered the operation appeared increasingly likely to succeed in her career. | Did the new doctor offer the operation? | Did the new doctor appear increasingly likely to succeed in her career? |
| 6 | The professor awarded the grant gained more attention from marine biologists. | Did the professor award the grant? | Did the professor gain more attention from marine biologists? |
| 7 | The technician refused the service stopped working almost immediately after the argument. | Did the technician refuse the service? | Did the technician stop working almost immediately after the argument? |
| 8 | The mechanic brought the truck needed several more hours to fully repair it. | Did the mechanic bring the truck? | Did the mechanic need several more hours to fully repair it? |
| 9 | The guitarist assigned the song failed dramatically because he never practiced enough. | Did the guitarist assign the song? | Did the guitarist fail dramatically because he never practiced enough? |
| 10 | The player paid the bonus remained essentially the same despite his sudden fame and wealth. | Did the player pay the bonus? | Did the player remain essentially the same despite his sudden fame and wealth? |
| 11 | The recent hire offered the job prepared many students for careers in media. | Did the recent hire offer the job? | Did the recent hire prepare many students for careers in media? |
| 12 | The assistant manager assigned the training seemed unnecessarily demanding to new staff. | Did the assistant manager assign the training? | Did the assistant manager seem unnecessarily demanding to new staff? |
| 13 | The mayor sent the document provided sufficient evidence that it was simply blackmail. | Did the mayor send the document? | Did the mayor provide sufficient evidence that it was simply blackmail? |
| 14 | The basketball player handed the contract created another controversy in the NBA. | Did the basketball player hand the contract? | Did the basketball player create another controversy in the NBA? |
| 15 | The engineer brought the equipment required constant supervision from senior technicians. | Did the engineer bring the equipment? | Did the engineer require constant supervision from senior technicians? |
| 16 | The little girl fed the lamb remained relatively calm despite having asked for beef. | Did the little girl feed the lamb? | Did the little girl remain relatively calm despite having asked for beef? |
| 17 | The yoga instructor offered the position demanded immense physical effort from everyone. | Did the yoga instructor offer the position? | Did the yoga instructor demand immense physical effort from everyone? |
| 18 | The governor awarded the contract received sweeping support across the entire state. | Did the governor award the contract? | Did the governor receive sweeping support across the entire state? |
| 19 | The patient refused the treatment continued causing uncomfortable scenes in the ER. | Did the patient refuse the treatment? | Did the patient continue causing uncomfortable scenes in the ER? |
| 20 | The operator brought the machine started working efficiently with the added automation. | Did the operator bring the machine? | Did the operator start working efficiently with the added automation? |
| 21 | The dancer assigned the ballet achieved incredible success for a new performer. | Did the dancer assign the ballet? | Did the dancer achieve incredible success for a new performer? |
| 22 | The contestant paid the money became unavailable and suddenly terminated his contract. | Did the contestant pay the money? | Did the contestant become unavailable and suddenly terminate his contract? |
| 23 | The new chef offered the restaurant separated mediocre cooks from gifted ones. | Did the new chef offer the restaurant? | Did the new chef separate mediocre cooks from gifted ones? |
| 24 | The apprentice baker assigned the oven produced smaller cakes because he lacked experience. | Did the apprentice baker assign the oven? | Did the apprentice baker produce smaller cakes because he lacked experience? |

---

## Notes

- The GP-mechanisms dataset pairs each ambiguous sentence (True) with a disambiguated version (False) that uses "that"-complementizers, commas, or "who was" relative clauses to remove the garden-path effect. The False rows are not included here but are available in the source CSV.
- The comprehension questions test whether the reader fell for the garden-path interpretation (Q_No) or parsed the intended meaning (Q_Yes).
- These could be useful as a multiview evaluation task: different parsing strategies / ambiguity resolution should produce distinct embedding signatures across views.
