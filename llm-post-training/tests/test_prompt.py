from llm_post_training.prompt import (
    RESPONSE_MARKER,
    build_example,
    build_prompt,
    extract_response,
)


def test_prompt_includes_instruction_and_marker():
    prompt = build_prompt("Summarize the text.", "")
    assert "Summarize the text." in prompt
    assert RESPONSE_MARKER in prompt
    assert "### Input:" not in prompt  # no context => no input block


def test_prompt_includes_context_when_present():
    prompt = build_prompt("Answer the question.", "The sky is blue.")
    assert "### Input:" in prompt
    assert "The sky is blue." in prompt


def test_build_example_appends_answer():
    record = {"instruction": "Say hi", "input": "", "output": "hello"}
    example = build_example(record)
    assert example.endswith("hello")
    assert RESPONSE_MARKER in example


def test_extract_response_roundtrip():
    record = {"instruction": "Say hi", "input": "", "output": "hello there"}
    example = build_example(record)
    assert extract_response(example) == "hello there"


def test_extract_response_without_marker_returns_stripped():
    assert extract_response("  bare text  ") == "bare text"
