from llm_post_training.config import DataConfig
from llm_post_training.data import (
    clean_dataset,
    clean_record,
    prepare,
    split_dataset,
)


def test_clean_record_drops_missing_required():
    assert clean_record({"instruction": "", "output": "x"}) is None
    assert clean_record({"instruction": "hi", "output": ""}) is None


def test_clean_record_normalises_whitespace():
    cleaned = clean_record({"instruction": "  do   this\n", "output": "  ok "})
    assert cleaned["instruction"] == "do this"
    assert cleaned["output"] == "ok"


def test_clean_dataset_dedupes_on_instruction_and_input():
    records = [
        {"instruction": "a", "input": "", "output": "1"},
        {"instruction": "a", "input": "", "output": "2"},  # dup key -> dropped
        {"instruction": "b", "input": "", "output": "3"},
    ]
    cleaned = clean_dataset(records, DataConfig())
    assert len(cleaned) == 2


def test_clean_dataset_respects_max_examples():
    records = [{"instruction": f"q{i}", "input": "", "output": "a"} for i in range(10)]
    cleaned = clean_dataset(records, DataConfig(max_examples=3))
    assert len(cleaned) == 3


def test_split_is_deterministic():
    records = [{"instruction": f"q{i}", "input": "", "output": "a"} for i in range(20)]
    cfg = DataConfig(eval_fraction=0.25, seed=7)
    train_a, eval_a = split_dataset(records, cfg)
    train_b, eval_b = split_dataset(records, cfg)
    assert eval_a == eval_b
    assert len(eval_a) == 5
    assert len(train_a) == 15


def test_prepare_writes_splits(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            f'{{"instruction": "q{i}", "input": "", "output": "a{i}"}}'
            for i in range(20)
        )
    )
    cfg = DataConfig(
        raw_path=str(raw),
        processed_dir=str(tmp_path / "processed"),
        eval_fraction=0.1,
    )
    stats = prepare(cfg)
    assert stats["cleaned"] == 20
    assert stats["train"] + stats["eval"] == 20
    assert (tmp_path / "processed" / "train.jsonl").exists()
    assert (tmp_path / "processed" / "eval.jsonl").exists()
