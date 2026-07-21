from llm_post_training.scoring import (
    aggregate,
    follows_instruction,
    token_f1,
)


def test_token_f1_identical():
    assert token_f1("hello world", "hello world") == 1.0


def test_token_f1_disjoint():
    assert token_f1("cats", "dogs") == 0.0


def test_token_f1_partial_between_zero_and_one():
    score = token_f1("the quick brown fox", "the slow brown fox")
    assert 0.0 < score < 1.0


def test_follows_instruction_requires_all_phrases():
    assert follows_instruction("Refund issued for $49", ["refund"]) is True
    assert follows_instruction("No action taken", ["refund"]) is False
    assert follows_instruction("", ["refund"]) is False


def test_aggregate_rolls_up():
    scores = [
        {"correct": True, "followed": True, "f1": 1.0},
        {"correct": False, "followed": True, "f1": 0.0},
    ]
    summary = aggregate(scores)
    assert summary["n"] == 2
    assert summary["correctness"] == 0.5
    assert summary["instruction_following"] == 1.0
    assert summary["mean_f1"] == 0.5


def test_aggregate_empty():
    summary = aggregate([])
    assert summary["n"] == 0
    assert summary["correctness"] == 0.0
