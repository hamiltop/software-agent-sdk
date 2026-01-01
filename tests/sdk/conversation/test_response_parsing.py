import pytest
from pydantic import BaseModel
from openhands.sdk.conversation.response_utils import parse_structured_response

class SimpleModel(BaseModel):
    value: str

class ComplexModel(BaseModel):
    name: str
    age: int

def test_parse_structured_response_raw_json():
    text = '{"value": "hello"}'
    result = parse_structured_response(text, SimpleModel)
    assert result.value == "hello"

def test_parse_structured_response_markdown_json():
    text = 'Here is the JSON:\n```json\n{"value": "hello"}\n```'
    result = parse_structured_response(text, SimpleModel)
    assert result.value == "hello"

def test_parse_structured_response_markdown_no_lang():
    text = '```\n{"value": "hello"}\n```'
    result = parse_structured_response(text, SimpleModel)
    assert result.value == "hello"

def test_parse_structured_response_embedded_json():
    text = 'Sure, here is the result: {"value": "hello"} I hope this helps.'
    result = parse_structured_response(text, SimpleModel)
    assert result.value == "hello"

def test_parse_structured_response_complex():
    text = '```json\n{"name": "Alice", "age": 30}\n```'
    result = parse_structured_response(text, ComplexModel)
    assert result.name == "Alice"
    assert result.age == 30

def test_parse_structured_response_invalid_json():
    text = '{"value": "hello"' # Missing closing brace
    with pytest.raises(ValueError):
        parse_structured_response(text, SimpleModel)

def test_parse_structured_response_missing_field():
    text = '{"other": "hello"}'
    with pytest.raises(ValueError):
        parse_structured_response(text, SimpleModel)

def test_parse_structured_response_heuristics():
    # Text with noise around JSON
    text = 'Some prefix { "value": "hello" } suffix'
    result = parse_structured_response(text, SimpleModel)
    assert result.value == "hello"

def test_parse_structured_response_noisy_braces_with_markdown():
    # If markdown is present, it should be prioritized over heuristic
    text = 'My { bad } value is ```json\n{"value": "ok"}\n```'
    result = parse_structured_response(text, SimpleModel)
    assert result.value == "ok"

def test_parse_structured_response_malformed_markdown_spaces():
    # Spaces around language identifier
    text = '``` json \n{"value": "hello"}\n```'
    result = parse_structured_response(text, SimpleModel)
    assert result.value == "hello"

def test_parse_structured_response_multiple_blocks():
    # Should find the first one
    text = 'First: ```{"value": "first"}``` Second: ```{"value": "second"}```'
    result = parse_structured_response(text, SimpleModel)
    assert result.value == "first"

