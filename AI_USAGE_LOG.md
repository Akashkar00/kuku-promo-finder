# AI usage log

Filled in during the build. The solution doc's "one place AI got it wrong" comes from here.

| When | Tool | Task | What it got wrong | How I caught / fixed it |
|---|---|---|---|---|
| 2026-11-20 | Claude Code (Opus 5.5) | Writing the show-brief prompt | Used two of this show's own speaker aliases as examples in a prompt meant to work on any show, which leaks test-show knowledge into the pipeline | Caught on review against the "no show-specific logic" rule; replaced with generic examples |
| 2026-11-20 | Claude Haiku 5.5 (pipeline) | Show brief from loglines | Called the antagonist a "stepsister" (loglines say "sister"), merged two separate clergy characters into one, and missed the script's descriptive speaker labels (labels that describe a character instead of naming them). Partly our prompt's fault: its example role was "cruel stepsister", and it never saw the speaker labels | Fed the input's own speaker-label list into the brief call, told it to copy relations exactly and keep uncertain labels separate, replaced the priming example |
| 2026-11-20 | Claude Code (Opus 5.5) | Writing the gold set and parser tests | Put verbatim script lines into files destined for the public repo (gold-set notes, a test fixture line), which would publish Kuku's confidential screenplay | Caught on a confidentiality pass before the first commit; paraphrased the notes, replaced the fixture line with an invented one, made the pack path an env var |
| 2026-11-20 | Claude Sonnet 5.5 (pipeline) | Mining promos | Split continuous stretches into back-to-back segments with 0 s between them; one pick became 4 "segments" and was rejected | Found while reading the run-1 eval misses; the assembler now merges segments closer than 1.5 s (test added) |
| 2026-11-21 | Claude Code (Opus 5.5) | Ranking evaluation | Concluded "self-scores don't discriminate" from an early-episode-weight test that was confounded: ties were broken by promo ID, which sorts by episode, re-creating the early-first order | Caught by an independent review pass (fresh-context reviewer agents); ties now break on the miner's own order, and MRR was added as a rank-sensitive check |
