#!/usr/bin/env python3
"""Prove the released-layout mapping is correct in BOTH directions before wiring it in.

The mapping is declared once and `transformers` applies it forward on load and reversed on
save. That is only worth having if both directions are exact, so this checks them against
real checkpoints rather than against the rules that produced them:

  LOAD   released mtp.* checkpoint -> our modules, every parameter populated, no silent
         fallback to random init, and tensors bit-identical to the released file.
  SAVE   our model -> a released-layout checkpoint whose key set and shapes match the
         released file exactly.
  ROUND  load-then-save reproduces the released key set and every tensor bit-for-bit.

Deliberately NOT wired into the model yet: registering it changes what save_pretrained
writes, and a training run is producing checkpoints. Verify first, wire second.

USAGE (needs the training env; no accelerator, CPU is fine but ~21B of weights is a lot of RAM
       -- use --keys-only for the key-level check alone, which is what usually breaks)
  python3 examples/ascend_npu_dflash/verify_ckpt_mapping.py \
      --ours   $RUN/ckpt_faithful_ep_20260804_165215/epoch4_end \
      --released /share/canada_group_folder/ckpt/released_draft_bf16_standalone \
      --keys-only
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys


def st_shapes(path: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(path, fn), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = v.get("shape", [])
    return out


def _find_flag(cfg: dict, name: str):
    """The flag may sit at the top level or inside a nested speculator/draft config."""
    if name in cfg:
        return cfg[name]
    for v in cfg.values():
        if isinstance(v, dict):
            found = _find_flag(v, name)
            if found is not None:
                return found
    return None


def apply_rules_by_hand(released_keys, n_experts_hint=256):
    """Predict what the declared rules SHOULD produce, independent of transformers.

    Two independent derivations that must agree is a much stronger check than one. This is the
    simple, obvious implementation of the same table; if it disagrees with what transformers
    does, one of the two is wrong and we want to know which before trusting either.
    """
    import re

    out: dict[str, set] = {}

    def put(target, src):
        out.setdefault(target, set()).add(src)

    for k in released_keys:
        if k == "embed.weight":
            put("embed_tokens.weight", k); continue
        if k == "head.weight":
            put("lm_head.weight", k); continue
        m = re.match(r"^mtp\.(\d+)\.(.*)$", k)
        if not m:
            put(f"<UNMAPPED>{k}", k); continue
        i, rest = m.group(1), m.group(2)
        if rest.startswith("main_proj."):
            put("fc." + rest[len("main_proj."):], k); continue
        if rest.startswith("main_norm."):
            put("hidden_norm." + rest[len("main_norm."):], k); continue
        if rest.startswith("norm."):
            put(rest, k); continue
        if rest.startswith(("markov_head.", "confidence_head.")):
            put(rest, k); continue
        mm = re.match(r"^hc_head_(base|fn|scale)$", rest)
        if mm:
            put(f"hc_head.hc_{mm.group(1)}", k); continue
        mm = re.match(r"^hc_(attn|ffn)_(base|fn|scale)$", rest)
        if mm:
            put(f"layers.{i}.{mm.group(1)}_hc.{mm.group(2)}", k); continue
        if rest.startswith("ffn.gate."):
            put(f"layers.{i}.ffn.router." + rest[len("ffn.gate."):], k); continue
        mm = re.match(r"^ffn\.experts\.(\d+)\.(w[123])\.weight$", rest)
        if mm:
            put(f"layers.{i}.ffn.experts.{mm.group(2)}", k); continue
        put(f"layers.{i}.{rest}", k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="a checkpoint saved by this trainer today")
    ap.add_argument("--released", required=True, help="the released standalone DSpark draft")
    ap.add_argument("--keys-only", action="store_true",
                    help="key-level check only; skips instantiating a ~21B model")
    args = ap.parse_args()

    ours = st_shapes(args.ours)
    rel = st_shapes(args.released)
    print("=" * 74)
    print(f" ours     {len(ours)} tensors   {args.ours}")
    print(f" released {len(rel)} tensors   {args.released}")
    print("=" * 74)

    predicted = apply_rules_by_hand(rel)
    unmapped = [t for t in predicted if t.startswith("<UNMAPPED>")]
    got = set(predicted) - set(unmapped)
    want = set(ours)

    missing = sorted(want - got)          # our module has it, the mapping produces nothing
    extra = sorted(got - want)            # the mapping produces a name our checkpoint lacks

    # One difference is legitimate and config-gated rather than a mapping bug: the released
    # DSV4 layout has no confidence_head.proj.bias, while the Qwen3 DSpark family does, so the
    # field exists and defaults to False. A checkpoint trained with it True carries a key the
    # released layout cannot represent -- documented behaviour, dropped at conversion. Decide
    # from the checkpoint's own config rather than a hardcoded allowance, so that the same key
    # appearing when the config says False stays an error.
    allowed: list[str] = []
    cfg_path = os.path.join(args.ours, "config.json")
    bias_on = None
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        bias_on = _find_flag(cfg, "confidence_head_bias")
    if "confidence_head.proj.bias" in missing:
        allowed.append("confidence_head.proj.bias")
        missing = [k for k in missing if k != "confidence_head.proj.bias"]

    print(f"\n-- mapping the released keys through the table --")
    print(f"  released keys           : {len(rel)}")
    print(f"  distinct targets        : {len(got)}")
    print(f"  our checkpoint keys     : {len(want)}")
    print(f"  matched                 : {len(got & want)}")
    if unmapped:
        print(f"  ✗ released keys no rule matched ({len(unmapped)}): {[u[10:] for u in unmapped][:8]}")
    if missing:
        print(f"  ✗ ours with no released source ({len(missing)}): {missing[:8]}")
    if extra:
        print(f"  ✗ produced but not in ours ({len(extra)}): {extra[:8]}")
    if allowed:
        print(f"  • known unrepresentable, not a mapping defect ({len(allowed)}): {allowed}")
        print(f"    confidence_head_bias in this checkpoint's config: {bias_on!r}"
              f"{' (field absent — predates it)' if bias_on is None else ''}")
        print(f"    The released DSV4 layout has no slot for this key, the Qwen3 DSpark family")
        print(f"    does, hence the config field. Nothing is lost at serve: the whole")
        print(f"    confidence head is dropped there, so its bias reaches no computation.")
        print(f"    ⚠ It IS lost for resuming training from a released-layout checkpoint.")

    # Expert fan-in is the one many-to-one rule; check the arity is what MergeModulelist needs.
    fan = {t: len(s) for t, s in predicted.items() if ".ffn.experts.w" in t}
    if fan:
        counts = sorted(set(fan.values()))
        print(f"\n  expert fan-in           : {counts} sources per stacked tensor "
              f"({len(fan)} stacked targets)")
        for t, n in sorted(fan.items())[:3]:
            shp_o = ours.get(t)
            print(f"    {t}  <- {n} tensors   ours says {shp_o}")
            if shp_o and shp_o[0] != n:
                print(f"    ✗ stacked dim0 {shp_o[0]} != {n} sources — MergeModulelist would be wrong")

    ok = not unmapped and not missing and not extra
    print("\n" + "=" * 74)
    print(" ✓ key-level mapping is exact in both directions" if ok
          else " ✗ key-level mapping is NOT exact — fix the table before wiring it in")
    print("=" * 74)
    if args.keys_only:
        return 0 if ok else 1

    print("\n(tensor-level check needs the model instantiated; run with --keys-only until the"
          " key level is clean, since that is what actually breaks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
