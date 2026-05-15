import pytest
from vocapp.logic import compare


def test_compare_exact_match():
    assert compare("gehen", "gehen") is True


def test_compare_case_insensitive():
    assert compare("Gehen", "gehen") is True


def test_compare_ignores_parentheses():
    assert compare("gehen (irregular)", "gehen") is True


def test_compare_umlauts():
    # depends on your normalization rules:
    assert compare("für", "fur") is False  # preserves äöü logic in your code


def test_compare_multiple_correct_order():
    assert compare("gehen, laufen", "laufen, gehen") is True


def test_compare_extra_spaces():
    assert compare(" gehen , laufen ", "laufen,gehen") is True


def test_compare_quotes_removed():
    assert compare('"gehen"', "gehen") is True
