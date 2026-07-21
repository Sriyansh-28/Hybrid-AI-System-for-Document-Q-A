# LLM Post-Training Pipeline for Domain Adaptation

This is a small, opinionated pipeline for taking an open 7B LLM (LLaMA-family)
and teaching it to follow instructions in *your* domain — support tickets,
internal docs, a specific product's Q&A, whatever you have data for. It uses
LoRA, so you get most of the quality of a full fine-tune at a fraction of the
GPU cost, and it's built so a teammate can re-run the whole thing without you
sitting next to them.

The pieces are deliberately separated: clean the data, train an adapter,
score it, merge it, serve it. Each step is one script and one config file, and
the config file is the complete record of what happened. Change a value, commit
it, and the experiment is reproducible.

## Why LoRA (the short version)

A full fine-tune updates all ~7B weights. That needs a lot of memory and a lot
of GPU hours. LoRA freezes the base model and only trains a small set of
low-rank matrices bolted onto the attention layers — usually well under 1% of
the parameters. In practice that's the difference between needing a cluster and
fitting on a single 24GB card (with 4-bit quantization, a.k.a. QLoRA), and it's
where the "~60% less compute" claim comes from. The trade-off is usually small
if the domain is well-scoped.

## Layout

```
llm-post-training/
├── configs/
│   ├── train.yaml          # all training + data hyperparameters
│   └── eval.yaml           # evaluation settings
├── data/
│   ├── raw/                # your raw instruction data goes here (JSONL)
│   └── processed/          # generated train/eval splits (gitignored)
├── scripts/                # the things you actually run
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── merge_and_export.py
│   └── serve.py
├── src/llm_post_training/  # the library the scripts call
│   ├── config.py           # YAML -> typed config objects
│   ├── data.py             # cleaning, dedup, splitting
│   ├── prompt.py           # the one true prompt template
│   ├── model.py            # tokenizer + base model + LoRA wiring
│   ├── train.py            # the SFT training loop
│   ├── scoring.py          # dependency-free scorers
│   ├── evaluate.py         # generate + score + write report
│   ├── export.py           # merge adapter into base weights
│   └── serve.py            # FastAPI serving layer
├── tests/                  # unit tests for the logic that doesn't need a GPU
├── Dockerfile              # serving image
├── requirements.txt        # full stack (training needs a GPU)
└── requirements-dev.txt    # just enough to run the tests on a laptop
```

## Data format

Raw data is JSONL, one instruction example per line. The fields:

```json
{
  "instruction": "Summarize the customer's issue.",
  "input": "Hi, my March invoice shows two $49 charges but I only have one plan...",
  "output": "The customer was double-charged $49 and wants a refund for the duplicate.",
  "required_phrases": ["refund"]
}
```

- `instruction` and `output` are required.
- `input` is optional context (leave it `""` if there isn't any).
- `required_phrases` is optional and only used at eval time — it's the
  instruction-following check ("did the answer actually mention the refund?").

There's a tiny `data/raw/sample_raw.jsonl` in here so you can run everything
end to end before you plug in your real 50K+ examples.

## Getting started

You can do everything except the actual GPU training on a laptop. Two install
options depending on what you're doing.

**Just working on the pipeline logic / running tests:**

```bash
pip install -r requirements-dev.txt
python -m pytest
```

**Full training + serving (needs a CUDA GPU):**

```bash
pip install -r requirements.txt
```

> Note: the base model (`meta-llama/Llama-2-7b-hf`) is gated on Hugging Face.
> Request access on the model page and run `huggingface-cli login` first, or
> point `model.base_model` in the config at any LLaMA-architecture model you
> already have.

## The workflow, step by step

### 1. Prepare the data

Cleans whitespace, drops empty/too-short examples, dedupes on the
instruction+input pair, then splits into train/eval deterministically (same
seed → same split, every time).

```bash
python scripts/prepare_data.py --config configs/train.yaml
```

You'll get a little summary of how many examples survived cleaning and where the
splits landed (`data/processed/train.jsonl` and `eval.jsonl`).

### 2. Train the adapter

```bash
python scripts/train.py --config configs/train.yaml
```

This loads the base model in 4-bit, attaches a fresh LoRA adapter, and runs
supervised fine-tuning over your formatted prompts. Only the adapter weights get
saved (to `outputs/adapter/` by default) — they're small, usually tens of MB.

Want to try different hyperparameters? Copy `configs/train.yaml`, change
`lora_r` or `learning_rate` or point `data.raw_path` at a different dataset, and
run again with `--config your_new_file.yaml`. Nothing in the code needs editing.

### 3. Evaluate

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

This is the part that replaces eyeballing a handful of outputs. It generates an
answer for every example in the eval split and scores each on:

- **Correctness** — token-overlap F1 against the reference answer, thresholded
  (default 0.75). Swap in embedding similarity or an LLM-as-judge here if you
  want something stricter.
- **Instruction-following** — did the answer contain the phrases the task
  required?

It prints headline numbers and writes a full per-example report to
`outputs/eval_report.json`. Run it on every candidate adapter and diff the
reports to catch regressions before they ship.

### 4. Merge for serving

Adapter-swapping is great if you're hosting many domains off one base model.
For a single-domain deployment it's simpler to bake the adapter into standalone
weights:

```bash
python scripts/merge_and_export.py \
  --config configs/train.yaml \
  --adapter outputs/adapter \
  --output outputs/merged
```

### 5. Serve

```bash
MODEL_PATH=outputs/merged python scripts/serve.py --host 0.0.0.0 --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Summarize the customer issue.", "context": "My invoice was double charged."}'
```

The serving layer here is a straightforward FastAPI wrapper — fine for internal
use and demos. For real throughput, serve the same merged weights with
[vLLM](https://github.com/vllm-project/vllm) or
[TGI](https://github.com/huggingface/text-generation-inference); the prompt
format is identical, so nothing else changes.

### Or with Docker

```bash
docker build -t llm-post-training .
docker run -p 8000:8000 -v $(pwd)/outputs/merged:/models/merged llm-post-training
```

## Reusing this for a different domain

That's the whole point of keeping it this generic. To adapt it:

1. Drop your JSONL into `data/raw/`.
2. Point `data.raw_path` at it in a config file.
3. Adjust `base_model` and the LoRA/training hyperparameters if you want.
4. Run steps 1–5.

No step needs the original author present, which is the idea — the config files
plus this README are the handoff.

## A note on the prompt template

The prompt lives in exactly one place (`src/llm_post_training/prompt.py`) and is
used identically at train, eval, and serve time. That's on purpose: a mismatch
between the string you trained on and the string you serve is one of the
sneakiest ways to get a model that looks great in evals and falls over in
production. If you change the template, you change it once and everything stays
in sync.

## Testing

The data cleaning, prompt formatting, scoring, and evaluation wiring are all
covered by unit tests that don't need a GPU or the model stack:

```bash
python -m pytest
```

The training and serving code paths pull in `torch`/`transformers`/`peft`
lazily, so importing and testing the rest works on any machine.
