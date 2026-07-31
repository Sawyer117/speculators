#!/usr/bin/env python3
"""Is the DUMPED training HS the REAL verifier HS? — the train/serve-gap root-cause probe.

Context: compare_train_vs_serve.py proved a train/serve gap that reduces (accept_len
freezes at first mismatch + DSpark is a single mask-token forward) to *dumped training
HS != eval-serve HS*. This script tests the dumped HS against baselines that are NOT a
second vLLM-Ascend dump (which we couldn't trust):

  MODE 1  SELF-CONSISTENCY (cheap, no 2nd serve, no HF) — baseline = the ROLLOUT TOKEN.
    The dumped FINAL hidden (`hidden_states[:, -1]`, already post-norm) through the
    verifier `lm_head` MUST argmax to the actual next token `token_ids[i+1]` IF the dump
    is the true verifier hidden and the rollout was greedy (temp=0). Mismatch rate:
      ~0%   -> dump is (argmax-)clean.
      ~11%  -> dump CORRUPTED / wrong-capture (== the train-serve slot0 gap 0.819-0.704).
    Run it on a conc=1 dump AND a conc=96 dump (the training over-subscription level):
    conc1~0% but conc96~11% => bf16 over-subscription garbage CONFIRMED as the root cause.
    (argmax-level: catches gross corruption; a subtle argmax-preserving norm shift passes -> use MODE 2.)

  MODE 2  HF REFERENCE (--hf-model) — baseline = an INDEPENDENT forward (transformers
    DeepSeek-V4, the only non-vLLM-Ascend oracle). Runs the model on the dumped
    `token_ids` and compares next-token argmax of `out.logits` (final norm + lm_head done
    inside HF, so NO fragile manual layer-indexing) against BOTH the rollout token and the
    dumped-final argmax. HF==rollout but dumped!=rollout => dumped is wrong.
    (Heavy: loads the full verifier. Optional; MODE 1 already nails the corruption case.)

Dump format (data.py:547-567): each ``hs_<idx>.safetensors`` = {
   "hidden_states": [seq_len, num_layers, hidden_size]  # [:, :-1]=aux [40,41,42], [:, -1]=final post-norm
   "token_ids":     [seq_len]                            # == input_ids
}
Get a dump: on the LIVE HS-dump (austin) serve, fire prefill-only requests so it writes
files (on_generate != delete, or grab before delete). One dir per concurrency level.

Usage:
  # inspect the real format first (verify my assumption on your files):
  python dsv4_hs_integrity_check.py --hs-dir ~/hs_conc1 --inspect
  # MODE 1 (lm_head only, ~1.8GB — light):
  python dsv4_hs_integrity_check.py --hs-dir ~/hs_conc1  --model-dir /home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16
  python dsv4_hs_integrity_check.py --hs-dir ~/hs_conc96 --model-dir /home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16
  # MODE 2 (+ HF independent forward — heavy):
  python dsv4_hs_integrity_check.py --hs-dir ~/hs_conc1 --model-dir <dir> --hf-model <dir>
"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file

# lm_head can be a dedicated weight or tied to the input embedding.
_LM_HEAD_KEYS = ("lm_head.weight", "model.embed_tokens.weight", "embed_tokens.weight")


def pick_device(want: str) -> str:
    if want != "auto":
        return want
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return "npu:0"
    except Exception:  # noqa: BLE001
        pass
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _pick_lm_head_key(keys, override: str | None):
    """DeepSeek-V4 may not call it 'lm_head.weight' — fuzzy-match: exact override, then the
    preferred names, then ANY '*lm_head*' key, then a tied 'embed_tokens.weight'."""
    keys = list(keys)
    if override:
        return override if override in keys else None
    for k in _LM_HEAD_KEYS:
        if k in keys:
            return k
    for k in keys:
        if "lm_head" in k.lower():
            return k
    for k in keys:
        if k.lower().endswith("embed_tokens.weight"):
            return k
    return None


def load_lm_head(model_dir: str, override_key: str | None):
    """Load ONLY the lm_head weight [vocab, H] from a (sharded) HF checkpoint — no full model."""
    import glob  # noqa: PLC0415

    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wmap = json.load(f)["weight_map"]
        k = _pick_lm_head_key(wmap.keys(), override_key)
        if k:
            with safe_open(os.path.join(model_dir, wmap[k]), framework="pt") as fh:
                return k, fh.get_tensor(k)
        head_like = [x for x in wmap if "head" in x.lower() or "embed" in x.lower()]
        raise SystemExit(f"lm_head not in index; head/embed keys = {head_like[:12]}; pass --lm-head-key")
    # no index -> scan shards (safe_open reads only the header, cheap)
    for p in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(p, framework="pt") as fh:
            k = _pick_lm_head_key(fh.keys(), override_key)
            if k:
                return k, fh.get_tensor(k)
    raise SystemExit(f"lm_head weight not found in {model_dir}; pass --lm-head-key")


def decile_rates(mism: torch.Tensor, n: int = 10) -> list[float]:
    """Mismatch rate per positional decile — the prompt->response transition self-locates
    (prompt tokens are user-given => high mismatch; response tokens are model-greedy => ~0)."""
    L = len(mism)
    if L == 0:
        return []
    out = []
    for d in range(n):
        a, b = L * d // n, L * (d + 1) // n
        seg = mism[a:b]
        out.append(seg.float().mean().item() if len(seg) else float("nan"))
    return out


def next_token_mismatch(pred_next: torch.Tensor, token_ids: torch.Tensor, min_pos: int):
    """pred_next[i] is the model's argmax AT position i (predicting token i+1). Compare to
    the actual token_ids[i+1]. Returns (mask_of_mismatch_over_valid, n_valid)."""
    p = pred_next[:-1]
    gt = token_ids[1:]
    valid = torch.arange(len(p)) >= min_pos
    mism = (p[valid] != gt[valid])
    return mism, int(valid.sum())


def logits_argmax(hidden: torch.Tensor, W: torch.Tensor, device: str, chunk: int) -> torch.Tensor:
    """argmax over vocab of hidden @ W.T, chunked over positions to bound memory."""
    preds = []
    Wt = W.float().to(device).t()  # [H, vocab]
    for s in range(0, hidden.shape[0], chunk):
        x = hidden[s:s + chunk].float().to(device)  # [c, H]
        preds.append((x @ Wt).argmax(-1).to("cpu"))
    return torch.cat(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs-dir", help="dir of hs_*.safetensors")
    ap.add_argument("--hs-files", nargs="*", help="explicit files (overrides --hs-dir)")
    ap.add_argument("--model-dir", help="verifier ckpt dir (for lm_head); required unless --inspect")
    ap.add_argument("--lm-head-key", help="override lm_head tensor key")
    ap.add_argument("--device", default="auto", help="auto|cpu|npu:0|cuda:0")
    ap.add_argument("--min-pos", type=int, default=0, help="skip the first N positions (prompt) in the headline rate")
    ap.add_argument("--chunk", type=int, default=256, help="positions per matmul chunk")
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--inspect", action="store_true", help="just print the dump format of the first file")
    ap.add_argument("--hf-model", help="MODE 2: HF model dir for an independent forward (heavy)")
    ap.add_argument("--hf-max-len", type=int, default=2048, help="truncate seq for the HF forward")
    args = ap.parse_args()

    files = args.hs_files or sorted(glob.glob(os.path.join(args.hs_dir or ".", "hs_*.safetensors")))
    if not files:
        raise SystemExit(f"no hs_*.safetensors found (hs-dir={args.hs_dir!r})")
    files = files[: args.max_files]

    if args.inspect:
        d = load_file(files[0])
        print(f"=== INSPECT {files[0]} ===")
        for k, v in d.items():
            print(f"  {k:28s} shape={tuple(v.shape)} dtype={v.dtype}")
        if "token_ids" in d:
            print(f"  token_ids[:12] = {d['token_ids'][:12].tolist()}")
        hs = d.get("hidden_states")
        if hs is not None and hs.ndim == 3:
            print(f"  -> num_layers={hs.shape[1]} (expect len([40,41,42])+1 = 4); "
                  f"final = hidden_states[:, -1] [T, {hs.shape[2]}]")
        print("If this matches [seq_len, num_layers, hidden_size] + token_ids, MODE 1 is ready.")
        return

    if not args.model_dir:
        raise SystemExit("--model-dir is required for the check (or use --inspect)")
    device = pick_device(args.device)
    key, W = load_lm_head(args.model_dir, args.lm_head_key)
    print(f"lm_head = '{key}' shape={tuple(W.shape)} | device={device} | files={len(files)}")

    hf_model = None
    if args.hf_model:
        print(f"[MODE 2] loading HF model {args.hf_model} (heavy)…")
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).eval()
        try:
            hf_model.to(device)
        except Exception as e:  # noqa: BLE001 — too big for one device; leave on CPU/meta-mapped
            print(f"  (HF .to({device}) failed: {e}; using whatever device_map loaded)")

    agg_mism, agg_n = 0, 0
    agg_hf_dumped, agg_hf_roll, agg_n_hf = 0, 0, 0
    per_file_deciles = []
    for path in files:
        d = load_file(path)
        hs, tok = d["hidden_states"], d["token_ids"].to(torch.long)
        final = hs[:, -1]  # [T, H] post-norm
        pred = logits_argmax(final, W, device, args.chunk)  # dumped-final next-token argmax
        mism, n = next_token_mismatch(pred, tok, args.min_pos)
        agg_mism += int(mism.sum()); agg_n += n
        per_file_deciles.append(decile_rates(next_token_mismatch(pred, tok, 0)[0]))

        line = f"  {os.path.basename(path):28s} T={len(tok):5d}  self-consistency mismatch={mism.float().mean():.3%} (n={n})"
        if hf_model is not None:
            ids = tok[: args.hf_max_len].unsqueeze(0).to(next(hf_model.parameters()).device)
            with torch.no_grad():
                logits = hf_model(input_ids=ids).logits[0].to("cpu")  # [t, vocab]
            hf_pred = logits.argmax(-1)  # HF next-token argmax (final norm + lm_head inside HF)
            t = hf_pred.shape[0]
            hf_vs_roll = (hf_pred[:-1] != tok[1:t]).float().mean().item()
            hf_vs_dumped = (hf_pred[:-1] != pred[:t - 1]).float().mean().item()
            agg_hf_roll += hf_vs_roll * (t - 1); agg_hf_dumped += hf_vs_dumped * (t - 1); agg_n_hf += (t - 1)
            line += f" | HF-vs-rollout={hf_vs_roll:.3%}  HF-vs-DUMPED={hf_vs_dumped:.3%}"
        print(line)

    print("=" * 78)
    print(f"AGGREGATE self-consistency mismatch (min_pos={args.min_pos}): "
          f"{agg_mism / max(agg_n, 1):.3%}  over {agg_n} positions, {len(files)} files")
    # mean decile curve — where mismatch drops = prompt->response boundary; read the RESPONSE (tail) deciles.
    if per_file_deciles:
        nd = max(len(x) for x in per_file_deciles)
        means = []
        for i in range(nd):
            vals = [x[i] for x in per_file_deciles if i < len(x) and x[i] == x[i]]
            means.append(sum(vals) / len(vals) if vals else float("nan"))
        print("  mismatch by positional decile (0%=start/prompt .. 100%=end/response):")
        print("   " + "  ".join(f"{m:.2%}" for m in means))
    if agg_n_hf:
        print(f"[MODE 2] AGGREGATE HF-vs-rollout={agg_hf_roll / agg_n_hf:.3%}  "
              f"HF-vs-DUMPED={agg_hf_dumped / agg_n_hf:.3%}")
    print("-" * 78)
    print("READ: MODE1 ~0% (esp. the response deciles) = dump argmax-clean. ~11% = corrupted")
    print("      (run conc=1 vs conc=96 dirs; a jump = over-subscription garbage).")
    print("      MODE2 HF-vs-rollout~0 but HF-vs-DUMPED high = dumped final != independent forward = dump wrong.")


if __name__ == "__main__":
    main()
