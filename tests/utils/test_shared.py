"""Tests for utils/ shared helpers."""

from utils.elements import serialize_elements
from utils.llm import strip_code_fence
from utils.url import normalise_url


def test_strip_code_fence():
    raw = '```json\n{"a": 1}\n```'
    assert strip_code_fence(raw) == '{"a": 1}'


def test_normalise_url_strips_query():
    assert normalise_url("http://localhost:5230/explore?q=1") == "http://localhost:5230/explore"


def test_serialize_elements_filters_missing_selector():
    data = serialize_elements([
        {"selector": "#a", "role": "button", "label": "Go"},
        {"role": "button", "label": "Skip"},
    ])
    assert "#a" in data
    assert "Skip" not in data or "#a" in data
