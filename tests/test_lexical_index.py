"""Tests for src.lexical_index."""

import pytest

from src.lexical_index import LexicalIndex


def test_basic_search():
    idx = LexicalIndex()
    idx.add(0, "def fetch_data(): pass")
    idx.add(1, "def process_data(): pass")
    idx.add(2, "class DataLoader: pass")

    results = idx.search("fetch data", top_k=5)
    assert len(results) >= 1
    assert results[0]["doc_id"] == 0


def test_remove_and_search():
    idx = LexicalIndex()
    idx.add(0, "hello world")
    idx.remove(0)
    results = idx.search("hello")
    assert results == []
