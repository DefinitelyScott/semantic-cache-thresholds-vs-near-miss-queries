"""A cheap rule-based check that runs after the similarity threshold.

The guard refuses to reuse a cached answer when the two queries disagree on
either of two things that a bag-of-words score cannot see well:

1. **Numbers**: the multiset of numbers mentioned (``2`` and ``two`` count as
   the same number).
2. **Negation**: whether the query contains a negation word.

It is deliberately simple. It is expected to help with the ``number`` and
``negation`` near-misses and to do nothing for role swaps, entity swaps or
antonyms; the experiment measures whether that expectation holds.
"""

from __future__ import annotations

from collections import Counter

from cache_eval.similarity import tokenize

NUMBER_WORDS: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}

NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "without", "none", "nor",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "cant", "wont", "shouldnt",
})


def numbers_in(text: str) -> Counter[str]:
    """Count the numbers mentioned in ``text``.

    Digits are read from inside tokens, so ``7pm`` gives ``7`` and ``$1500``
    gives ``1500``. Small number words (one to twelve) are mapped to digits.
    """
    found: Counter[str] = Counter()
    for token in tokenize(text):
        if token in NUMBER_WORDS:
            found[NUMBER_WORDS[token]] += 1
            continue
        digits = "".join(ch if ch.isdigit() else " " for ch in token).split()
        found.update(digits)
    return found


def is_negated(text: str) -> bool:
    """True if ``text`` contains any word from ``NEGATION_WORDS``."""
    return any(token in NEGATION_WORDS for token in tokenize(text))


def guard_allows(cached: str, incoming: str) -> bool:
    """Return True if the guard would let ``incoming`` reuse ``cached``'s answer."""
    return numbers_in(cached) == numbers_in(incoming) and is_negated(cached) == is_negated(incoming)
