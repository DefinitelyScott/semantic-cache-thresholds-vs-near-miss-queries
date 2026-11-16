from cache_eval.guard import guard_allows, is_negated, numbers_in


def test_numbers_are_read_from_digits_inside_tokens_and_number_words():
    assert numbers_in("Book a table for 2 at 7pm") == {"2": 1, "7": 1}
    assert numbers_in("a table for two at 7 this evening") == {"2": 1, "7": 1}
    assert numbers_in("laptops under $1500") == {"1500": 1}
    assert numbers_in("no numbers here") == {}


def test_negation_detection():
    assert is_negated("Which cereals do not contain gluten?")
    assert is_negated("flights that don't have wifi")
    assert is_negated("hotels without free cancellation")
    assert is_negated("Recipes that use no eggs")
    assert not is_negated("Which cereals contain gluten?")
    # "know" contains "no" but is not the word "no".
    assert not is_negated("I know the answer")


def test_guard_blocks_number_and_negation_changes():
    assert not guard_allows("Set a timer for 10 minutes", "Set a timer for 30 minutes")
    assert not guard_allows("Recipes that use eggs", "Recipes that use no eggs")


def test_guard_allows_paraphrases_that_spell_numbers_differently():
    assert guard_allows("Book a table for 2 people at 7pm tonight",
                        "Can I reserve a table for two at 7 this evening?")


def test_guard_cannot_see_role_swaps_or_antonyms():
    # Documented limitation: these pass the guard and must be caught some other way.
    assert guard_allows("Convert US dollars to euros", "Convert euros to US dollars")
    assert guard_allows("How do I zoom in on a map?", "How do I zoom out on a map?")
