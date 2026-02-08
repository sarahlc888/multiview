# Task: Add semantic filter criteria for HackerNews

## Goal
Add "semantic filter" criteria to HackerNews that go beyond topic — capturing the *vibe* and *appeal type* of posts.

## Type
ADD_CRITERIA

## Current state
- Docset exists: `src/multiview/docsets/hackernews.py` (julien040/hacker-news-posts from HuggingFace)
- Existing criteria in `configs/available_criteria.yaml` under `hackernews:`:
  - `angle_of_interest` — what motivates someone to click (has very detailed tag hints)
  - `article_topic` — subject matter

## What to add
From Task B notes, the "semantic filter" idea:
- **fun_factor**: "hacker news by fun-factor" — is this post fun/entertaining vs. serious/dense?
- **personal_blog**: "hacker news by personal blog" — is this a personal blog post vs. corporate/institutional content?

The existing `angle_of_interest` criterion already captures a lot of the "vibe" dimension with tags like `fun_toy_project`, `browser_game`, `whimsical_gimmick`, `personal_project_white_whale`. Consider whether new criteria are truly orthogonal or if `angle_of_interest` already covers them.

## Reference files
- `configs/available_criteria.yaml` — hackernews section (~line 132)
- `src/multiview/docsets/hackernews.py`

## Original design notes (verbatim from b-add_data.md)
```
Look at the top 100 stories per day
Embed everything

But based on what?
If you just embed the headlines, you can probably get various groups

People are kind of interested in AI stuff
People are interested in space, medicine, etc

But there's also a particular type of vibe WRT hacker news itself - like, why is it on hacker news

And I feel like this will be cool and higher level

we need to be able to toggle between these views
```

## Steps
- [ ] Evaluate whether `angle_of_interest` already subsumes fun_factor and personal_blog
- [ ] If new criteria are needed, add them to `configs/available_criteria.yaml` under `hackernews:`
- [ ] Run eval with the semantic filter criteria
- [ ] The broader vision is a "toggle between views" UI — note this as a future visualizer task

## Exit criteria
- [ ] Decision documented on whether new criteria needed vs. existing coverage sufficient
- [ ] If new criteria added, eval runs exist in `outputs/`
