#!/usr/bin/env python3
"""Eval the served draft on a TRAINING-rollout sample — WITHOUT modifying Evaluator.py.

Imports the proven ``Evaluator.py`` as a module and **injects** a ``trainsample``
dataset at runtime (Evaluator is never edited — team rule: don't touch the working
eval script). Purpose = the train/serve-mismatch test: measure SERVE accept_len on
the SAME prompt distribution as the training metric (~3.57), vs the harder gsm8k
``--dataset all``:
  * serve ≈ train on this sample -> NO serve mismatch; the eval gap is data
    difficulty (gsm8k harder than rollout) -> lever = data/recipe.
  * serve << train on this sample -> serve mismatch confirmed -> hunt the bug.

Build the sample first:
  python build_train_eval_sample.py --in <rollout.jsonl> --out ~/train_sample_500.jsonl --n 500
Then (serve already up on :7000, same serve as the ep1end eval):
  TOKENIZER=$TOK SAMPLE_FILE=~/train_sample_500.jsonl CONCURRENCY=48 \
    python eval_trainsample.py | tee ~/eval_trainsample.txt
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import Evaluator  # noqa: E402  — the proven client, imported UNMODIFIED


def _tokenizer_default() -> str:
    if os.environ.get("TOKENIZER"):
        return os.environ["TOKENIZER"]
    for d in (
        "/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16",
        "/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16",
        "/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16",
    ):
        if os.path.isdir(d):
            return d
    return "/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16"


def main() -> None:
    sample = os.environ.get("SAMPLE_FILE")
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sample = sys.argv[1]
    if not sample or not os.path.isfile(sample):
        sys.exit(
            f"SAMPLE_FILE not found: {sample!r}\n"
            "Build it first: python build_train_eval_sample.py --in <rollout.jsonl> "
            "--out ~/train_sample_500.jsonl --n 500"
        )

    # Inject the dataset into the imported module — Evaluator.py itself is NOT edited.
    Evaluator.DATASETS["trainsample"] = {
        "load_args": ("json",),
        "load_kwargs": {"data_files": sample, "split": "train"},
        "format": lambda x: x["prompt"],
    }

    # corp proxy hijacks localhost; datasets/hub offline (local json needs no network)
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("OFFLINE", "1"))
    os.environ.setdefault("HF_DATASETS_OFFLINE", os.environ.get("OFFLINE", "1"))

    port = os.environ.get("PORT", "7000")
    argv = [
        "Evaluator.py",
        "--base-url", f"http://localhost:{port}",
        "--model", os.environ.get("MODEL", "dsv4"),
        "--tokenizer", _tokenizer_default(),
        "--dataset", "trainsample",
        "--concurrency", os.environ.get("CONCURRENCY", "48"),
        "--warmup-steps", os.environ.get("WARMUP", "10"),
        "--max-new-tokens", os.environ.get("MAX_NEW", "2048"),
        "--temperature", "0", "--top-p", "1", "--top-k", "1",
    ]
    if os.environ.get("KEEP_WARMUP", "1") == "1":
        argv += ["--keep-warmup-samples"]
    ct = os.path.join(_HERE, "dsv4_chat_template.jinja")
    if os.path.isfile(ct):
        argv += ["--chat-template", ct]
    np = os.environ.get("NUM_PROMPTS")
    if np and np.isdigit() and int(np) > 0:
        argv += ["--num-prompts", np]

    print(f">>> [trainsample] serving-eval on {sample} via :{port} (Evaluator.py unmodified)")
    sys.argv = argv
    Evaluator.main()


if __name__ == "__main__":
    main()
