import pytest

from llm_post_training.config import load_config


def test_load_config_reads_overrides(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "model:\n"
        "  base_model: my-org/my-model\n"
        "  lora_r: 8\n"
        "train:\n"
        "  epochs: 1.0\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.model.base_model == "my-org/my-model"
    assert cfg.model.lora_r == 8
    assert cfg.train.epochs == 1.0
    # Untouched fields keep their defaults.
    assert cfg.model.lora_alpha == 32


def test_load_config_rejects_unknown_keys(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("train:\n  notarealkey: 5\n")
    with pytest.raises(ValueError):
        load_config(cfg_file)
