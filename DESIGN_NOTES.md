# Design notes

Why the experiment is built the way it is, what was traded away, and what I
would do next. Written so every choice can be explained and defended.

## 1. The question

A similarity cache trades correctness for cost: every cache hit saves an LLM
call, and every wrong hit returns an answer to a question nobody asked. The
risk lives entirely in the threshold, so the real question is how the
threshold gets validated.

The common check is "pick a threshold where unrelated questions don't match."
That check has an obvious blind spot: the questions most likely to be served
wrongly are not unrelated ones. They are questions that differ from a cached
one by a single number, word or word order. So the experiment scores the same
cached questions against both kinds of negative and compares.

I framed it as an **evaluation-method** question (easy negatives vs. hard
negatives) rather than "which similarity measure is best," because the method
point holds for any measure, including embeddings, and it is the part a team
can act on immediately: add near-misses to the test set.

## 2. Why triples

Each cached question comes with exactly one paraphrase and one near-miss.
That pairing allows a **per-anchor comparison**: for each cached question,
did the near-miss outscore the paraphrase? That is a sharper statement than
comparing two pooled score distributions, because it holds the cached
question fixed. It also gives a simple exact test (a sign test on 30 pairs).

The unrelated question is not written separately; it is the paraphrase of the
question 15 places away. That keeps the data file small, guarantees the
unrelated question is a natural-sounding query (it is someone's paraphrase),
and, because the list is grouped by type in blocks of 6, guarantees it comes
from a different near-miss type.

## 3. The five near-miss types

I picked types that each break a specific property of word-overlap
similarity:

| type | what it tests |
|---|---|
| number | one token changes, and it is the token the answer depends on |
| negation | one short, common word is added; low IDF weight, huge meaning change |
| role swap | identical words in a different order; invisible to any bag of words |
| entity | one rare word changes (Austria / Australia) |
| antonym | one word flips polarity (increase / decrease) |

Six per type is enough to see whether a mitigation affects a type at all
(0/6 vs. 6/6), not enough to estimate rates within a type precisely.

## 4. Similarity measures

- **TF-IDF cosine (main measure).** The simplest score with a sensible notion
  of "this word matters more" and the closest lexical analogue of an
  embedding cosine. Implemented in about 30 lines, so every number can be
  traced by hand.
- **Jaccard.** A cruder baseline to show the result is not specific to IDF
  weighting.
- **TF-IDF with word pairs.** Added after seeing that role swaps score exactly
  1.0. The obvious objection is "that is just bag-of-words being order-blind";
  word pairs make the score order-aware, so this tests the objection
  directly. It does not help: paraphrases share even fewer word pairs than
  words, so they fall further than the near-misses.
- **No stopword removal.** Negation words are classic stopwords. Removing them
  would make negation near-misses score nearly identical to the cached question,
  which would exaggerate the failure for reasons unrelated to the question.
- **Why not scikit-learn's `TfidfVectorizer`?** It would work, and I used its
  default smoothed IDF formula so results match. Writing it out keeps the
  project dependency-free and makes the tokenizer choices (possessives,
  contractions, `$800`) explicit and testable.

## 5. How the threshold is "tuned"

`lowest_safe_threshold` picks the lowest value on a 0.05 grid at which no
negative is served. This mimics a sensible manual process: raise the
threshold until the bad examples stop. Two consequences, both documented:

- On the unrelated set the tuned TF-IDF threshold (0.20) sits just above the
  closest unrelated pair (0.1999). A different grid would move it slightly,
  but the near-miss result does not depend on it: near-misses are served at
  every threshold up to 0.65.
- On the near-miss set there is often **no** safe threshold, because role
  swaps score 1.0. The function returns `None` rather than inventing one, and
  the report prints "none".

Scores are compared with a 1e-9 tolerance (`metrics.EPS`) because two
identical word bags can produce a cosine of 0.9999999999999998, which would
otherwise fail a `>= 1.0` check and hide the role-swap result.

## 6. The guard

The guard is a post-check, not a new similarity score: after the threshold
says "close enough," it refuses reuse if the numbers differ or exactly one of
the two questions contains a negation word. I chose those two checks because
they are cheap, explainable, and target two of the five types; the other
three were left alone on purpose, so the experiment can show what a targeted
rule does and does not cover.

The most important property to check was that the guard does not block
paraphrases. "Book a table for 2" vs. "a table for two" would fail a naive
digit check, so number words one to twelve are mapped to digits. The test
`test_guard_never_blocks_a_paraphrase` pins that 0/30 paraphrases are blocked.

## 7. Honesty about what the result is

The lexical result is nearly guaranteed by how near-misses are defined, and
the README says so first in its limitations. What I think is worth showing:

1. The size of the gap between the two evaluations on the *same scores*
   (AUC 0.999 vs. 0.020).
2. That no threshold and no order-aware variant rescues a lexical cache.
3. That a targeted guard removes exactly the types it targets, with zero
   collateral damage on paraphrases in this set.

The embedding script exists so the real open question (do embedding caches
show the same inversion, and for which types?) can be answered with one
command. I have not run it, so there are no embedding numbers anywhere in the
repo.

## 8. Testing approach

- **Unit tests per module** with tiny inputs whose answers are obvious by hand
  (for example, the IDF formula on a three-document corpus, AUC on four
  scores, the sign test against hand-computed binomial values).
- **Experiment logic** tested with a fake similarity function that returns
  fixed scores, so threshold tuning and guard counts can be checked without
  depending on the real data.
- **Pinned README numbers** in `tests/test_reported_numbers.py`. If code or
  data changes a reported number, the build fails until the README is
  updated.
- **Smoke tests** run the real scripts in a subprocess and draw the figure
  into a temporary folder.

## 9. Alternatives considered

- **Use a public paraphrase dataset (Quora Question Pairs, PAWS).** PAWS is
  built from word-swap pairs and would be a great source of hard negatives,
  but it needs a download, it is not about cache-style requests, and its
  labels are about meaning rather than "would the same answer be correct."
  Writing 30 triples kept the repo offline and the labels unambiguous.
- **Run a local sentence-embedding model by default.** That would answer the
  embedding question offline, but it adds a large dependency (PyTorch or
  ONNX Runtime plus model weights) to a repo meant to be reviewed in 15
  minutes. It is the first item under "next".
- **Report precision/recall at many thresholds instead of a tuned point.**
  The full threshold table is printed by the script; the README leads with
  the tuned point because that is the decision a team actually makes.

## 10. What I would do next

1. Run `scripts/run_embedding_eval.py` with one or two embedding models and
   report per-type results. My expectation, untested: embeddings will handle
   some entity swaps and antonyms better than TF-IDF, and still struggle with
   role swaps and numbers.
2. Add an entity check to the guard (cached and incoming capitalized words must
   match) and measure it the same way, including its paraphrase cost.
3. Grow the set with near-misses taken from a real query log, where the type
   mix is not chosen by me.
4. Measure the cost side: a cache's value is hit rate times cost per call, so
   report "calls saved" next to "wrong answers served" for each threshold.
