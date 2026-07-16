#!/usr/bin/env python3
"""Assemble a STANDALONE vllm-ascend draft dir from the RELEASED fp8 DSpark model.

The released `DeepSeek-V4-Flash-DSpark` ships the draft BUNDLED into the 48-shard fp8 model:
the draft is the `mtp.*` tensors (last 3 shards, fp8 e4m3 block-quant + `.scale`), and it SHARES
the target's top-level `embed.weight` (shard 1) + `head.weight` (shard 45) — it has no embed/head
of its own. vllm's draft `load_weights` wants exactly {mtp.* , embed.weight , head.weight}.

This builds that standalone dir WITHOUT needing shards 1/45: it takes `mtp.*` (fp8, verbatim — no
dequant, pure byte copy) from the released model, and BORROWS `embed.weight` + `head.weight` from
our already-serving converted bf16 draft dir (identical base model / vocab, embed is unquantized).
config.json = our proven-good bf16 draft config + the released fp8 `quantization_config` grafted in
(the ONLY change needed so vllm loads the mtp linears as fp8; embed/head stay bf16, no scale => skipped).

  python scripts/build_released_draft_dir.py \
      --released   /path/DeepSeek-V4-Flash-DSpark     `# has shards w/ mtp.* + config.json + index` \
      --our-draft  /share/.../ckpt_faithful_ep1_vllm  `# our converted bf16 draft (embed/head source)` \
      --out        /share/.../released_draft_fp8_standalone

Serve it on our bf16 harness (isolates: known-good draft, our exact target + aux path):
  DRAFT=<out> MODEL=/share/.../DeepSeek-V4-Flash-bf16 NUM_SPEC=5 bash serve_dsv4_bf16_dualnode.sh head
  (put <out> on /share so BOTH nodes see it, like the converted draft.)

Runs on CPU (pure I/O, no fp8 arithmetic). Needs torch (>=2.1 for float8_e4m3fn) + safetensors (>=0.4).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def _load_index(d: Path) -> dict:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    # single-file fallback: map every key to the one file
    from safetensors import safe_open  # noqa: PLC0415

    single = d / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise SystemExit(f"!! no model.safetensors[.index.json] in {d}")


def _tensor(d: Path, wm: dict, key: str):
    """Load a single tensor by key using the dir's weight_map (verbatim, no cast)."""
    from safetensors.torch import load_file  # noqa: PLC0415

    if key not in wm:
        raise SystemExit(f"!! '{key}' not found in {d} (keys sample: {list(wm)[:3]})")
    sd = load_file(str(d / wm[key]))  # small: only the shard holding `key`
    return sd[key]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--released", required=True, help="released fp8 DSpark model dir (mtp.* source)")
    ap.add_argument("--our-draft", required=True, dest="ours",
                    help="our converted bf16 draft dir (embed.weight + head.weight source)")
    ap.add_argument("--out", required=True, help="output standalone draft dir")
    ap.add_argument("--ignore-embed-head", action=argparse.BooleanOptionalAction, default=True,
                    help="add embed/head to the fp8 quant_config's `ignored_layers` so vllm does NOT "
                         "try to fp8-quantize the borrowed bf16 embed/head (they have no .scale). "
                         "Default ON (pre-empts the fp8 load hiccup); --no-ignore-embed-head to disable.")
    args = ap.parse_args()

    from safetensors.torch import save_file  # noqa: PLC0415

    rel, ours, out = Path(args.released), Path(args.ours), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rel_wm = _load_index(rel)
    ours_wm = _load_index(ours)

    # --- 1) collect the fp8 mtp.* tensors (verbatim), loading each source shard ONCE ---
    mtp_keys = sorted(k for k in rel_wm if k.startswith("mtp."))
    if not mtp_keys:
        raise SystemExit(f"!! no mtp.* keys in {rel}/index — is this the released DSpark model?")
    shards = sorted({rel_wm[k] for k in mtp_keys})
    print(f">>> released: {len(mtp_keys)} mtp.* tensors across {len(shards)} shard(s): {shards}")
    from safetensors.torch import load_file  # noqa: PLC0415

    merged: dict = {}
    n_scale = 0
    for sh in shards:
        sd = load_file(str(rel / sh))
        for k, v in sd.items():
            if k.startswith("mtp."):
                merged[k] = v
                n_scale += k.endswith(".scale")
    print(f"    kept {len(merged)} mtp.* ({n_scale} .scale => fp8 block-quant confirmed)")

    # --- 2) borrow embed.weight + head.weight from our bf16 draft ---
    for k in ("embed.weight", "head.weight"):
        merged[k] = _tensor(ours, ours_wm, k)
        print(f"    borrowed {k}: {tuple(merged[k].shape)} {merged[k].dtype}  (from our bf16 draft)")

    # sanity: layer count + head-level placement should mirror the released reference
    lids = sorted({int(m.group(1)) for k in merged for m in [re.match(r"mtp\.(\d+)\.", k)] if m})
    assert "mtp.0.main_proj.weight" in merged, "main_proj must sit on mtp.0"
    last = lids[-1]
    assert f"mtp.{last}.markov_head.markov_w1.weight" in merged, f"markov_head must sit on mtp.{last}"
    print(f">>> mtp layers {lids}; main_proj@0, heads@{last}  (matches released layout)")

    # --- 3) write weights (single file; ~13GB fp8 + ~2GB bf16 embed/head is well within limits) ---
    save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f">>> wrote {out/'model.safetensors'} ({len(merged)} tensors)")

    # --- 4) config.json = our proven-good draft config + released fp8 quantization_config ---
    ours_cfg = json.loads((ours / "config.json").read_text())
    rel_cfg = json.loads((rel / "config.json").read_text())
    qcfg = rel_cfg.get("quantization_config")
    if qcfg is None:
        raise SystemExit(f"!! released config has no quantization_config — is {rel} the fp8 release?")
    qcfg = dict(qcfg)  # copy — don't mutate rel_cfg
    if args.ignore_embed_head:
        ig = list(qcfg.get("ignored_layers", []))
        for name in ("embed", "head"):  # borrowed bf16, no .scale => must stay unquantized
            if name not in ig:
                ig.append(name)
        qcfg["ignored_layers"] = ig
    ours_cfg["quantization_config"] = qcfg
    (out / "config.json").write_text(json.dumps(ours_cfg, indent=2))
    print(f">>> wrote config.json (base=our bf16 draft, grafted fp8 quant: {qcfg.get('quant_method')}"
          f"{', ignored_layers=' + str(qcfg['ignored_layers']) if args.ignore_embed_head else ''})")

    # copy small companions if present (harmless; draft_model_config mostly ignores tokenizer)
    for name in ("generation_config.json",):
        src = ours / name
        if src.exists():
            shutil.copy2(src, out / name)

    # --- verify embed/head geometry vs config ---
    V, H = ours_cfg.get("vocab_size"), ours_cfg.get("hidden_size")
    for k in ("embed.weight", "head.weight"):
        assert tuple(merged[k].shape) == (V, H), f"{k} {tuple(merged[k].shape)} != ({V},{H})"
    print(f">>> embed/head geometry OK vs config (vocab={V}, hidden={H})")
    print("\n✓ standalone released fp8 draft dir ready:", out)
    print("  next:  DRAFT=%s MODEL=/share/.../DeepSeek-V4-Flash-bf16 NUM_SPEC=5 \\" % out)
    print("         bash examples/ascend_npu_dflash/serve_dsv4_bf16_dualnode.sh head")


if __name__ == "__main__":
    main()
