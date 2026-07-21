from llm_post_training.config import PipelineConfig
from llm_post_training.evaluate import run_evaluation, score_predictions


def test_score_predictions_perfect():
    records = [
        {"instruction": "q", "output": "the answer is 42", "required_phrases": ["42"]},
    ]
    result = score_predictions(records, ["the answer is 42"], similarity_threshold=0.75)
    assert result["summary"]["correctness"] == 1.0
    assert result["summary"]["instruction_following"] == 1.0


def test_score_predictions_wrong_answer():
    records = [
        {"instruction": "q", "output": "the answer is 42", "required_phrases": ["42"]},
    ]
    result = score_predictions(records, ["totally unrelated"], similarity_threshold=0.75)
    assert result["summary"]["correctness"] == 0.0
    assert result["summary"]["instruction_following"] == 0.0


def test_run_evaluation_with_stub_generator(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        '{"instruction": "Say hi", "input": "", "output": "hello", "required_phrases": ["hello"]}\n'
    )
    cfg = PipelineConfig()
    cfg.eval.eval_path = str(eval_path)
    cfg.eval.report_path = str(tmp_path / "report.json")

    # Stub generator that always returns the reference answer.
    report = run_evaluation(cfg, generate=lambda prompt: "hello")

    assert report["summary"]["correctness"] == 1.0
    assert (tmp_path / "report.json").exists()
