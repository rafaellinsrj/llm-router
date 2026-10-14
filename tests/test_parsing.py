import pytest

from llm_router import InvalidJSON
from llm_router.parsing import parse_json


def test_plain_and_fenced():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Here:\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_light_repair_trailing_comma_and_truncation():
    assert parse_json('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}
    assert parse_json('{"a": {"b": 1') == {"a": {"b": 1}}


@pytest.mark.parametrize("text", ["", "no json", "[1, 2]", "{}"])
def test_invalid(text):
    with pytest.raises(InvalidJSON):
        parse_json(text)
