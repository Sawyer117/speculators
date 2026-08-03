#!/usr/bin/env python3
"""Name-agnostic WEIGHT fingerprint match between two checkpoints.

WHY: the forward-parity harness shows a precision-invariant, uniform, cosine≈0.975
divergence — the textbook signature of "the --ckpt weights are NOT the ones the serve
actually ran" (a forward CODE bug is either near-perfect or cosine≈0, not a stable 0.975).
This tool decides it directly: are dir A and dir B the SAME trained weights?

It does NOT rely on tensor NAMES (the train ckpt has 84 consolidated tensors; the converted
serve draft has ~2378 with per-expert splits + renamed mtp.* keys). Instead it fingerprints
each tensor by VALUE — (numel, round(sum), round(abs.sum), round(abs.max)) — and reports how
many of A's fingerprints appear in B. The 1:1 non-expert tensors (fc/main_proj, norms, MLA
q/kv/o, markov, mHC head, hidden_norm) are bit-exact across the train↔serve convert (verified
2378/2378), so they MUST match if it's the same trained state. Consolidated MoE-expert tensors
(a few big [n_experts,·] blocks) won't line up 1:1 with the split serve experts — expected; the
non-expert match rate is the signal.

READ:
  * non-expert match rate ~100%  → SAME weights. The parity divergence is NOT a ckpt mismatch.
  * match rate low / near 0      → DIFFERENT weights (wrong epoch subdir, or the serve loaded a
                                   draft converted from another run) → THAT is the ~13 gap. Point
                                   --ckpt at the epoch subdir that DOES match and re-run parity.

USAGE (run for each candidate epoch subdir vs the served draft dir):
  python dsv4_weight_fingerprint_match.py \
      /home/a00652497/dspark_austin/run/ckpt_faithful_ep_20260729_092941/1 \
      /share/canada_group_folder/ckpt/dsv4_dspark_drafts/<the served draft name>
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import glob
import os
import sys


def _iter_safetensors(path):
    """Yield (key, tensor) over every .safetensors shard in a dir (or a single file)."""
    from safetensors import safe_open
    files = ([path] if path.endswith(".safetensors")
             else sorted(glob.glob(os.path.join(path, "*.safetensors"))))
    if not files:
        # fall back to a torch .bin / pytorch_model
        import torch
        bins = ([path] if path.endswith(".bin")
                else sorted(glob.glob(os.path.join(path, "*.bin"))))
        if not bins:
            sys.exit(f"no .safetensors or .bin under {path}")
        for b in bins:
            sd = torch.load(b, map_location="cpu")
            for k, v in sd.items():
                yield k, v
        return
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                yield k, sf.get_tensor(k)


def _fingerprint_dir(path):
    """dir -> (dict fp->count, n_tensors, n_expert_like). fp = value signature."""
    fps = {}
    n = 0
    n_expert = 0
    for k, t in _iter_safetensors(path):
        n += 1
        tf = t.detach().float()
        s = round(float(tf.sum().item()), 2)
        a = round(float(tf.abs().sum().item()), 2)
        m = round(float(tf.abs().max().item()), 4) if tf.numel() else 0.0
        fp = (tf.numel(), s, a, m)
        fps[fp] = fps.get(fp, 0) + 1
        # heuristic: an "expert-block" tensor has a leading dim that looks like an expert count
        if t.dim() >= 2 and t.shape[0] in (256, 128, 64, 32):
            n_expert += 1
    return fps, n, n_expert


def _small_fps(path, max_numel):
    """FAST probe: value-fingerprint ONLY the small tensors (numel <= max_numel), reading
    each tensor's SHAPE from the safetensors header first so the huge per-expert MoE blocks
    are skipped without ever loading them. norms/biases/proj/markov/mHC are plenty to identify
    a training state. Returns dict fp->count."""
    from safetensors import safe_open
    files = ([path] if path.endswith(".safetensors")
             else sorted(glob.glob(os.path.join(path, "*.safetensors"))))
    if not files:
        # .bin fallback: no cheap shape peek, load it all (rare path)
        fps = {}
        for k, t in _iter_safetensors(path):
            if 0 < t.numel() <= max_numel:
                tf = t.detach().float()
                fp = (t.numel(), round(float(tf.sum()), 2),
                      round(float(tf.abs().sum()), 2), round(float(tf.abs().max()), 4))
                fps[fp] = fps.get(fp, 0) + 1
        return fps
    fps = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                shape = sf.get_slice(k).get_shape()
                numel = 1
                for d in shape:
                    numel *= d
                if numel == 0 or numel > max_numel:
                    continue
                tf = sf.get_tensor(k).detach().float()
                fp = (numel, round(float(tf.sum().item()), 2),
                      round(float(tf.abs().sum().item()), 2),
                      round(float(tf.abs().max().item()), 4))
                fps[fp] = fps.get(fp, 0) + 1
    return fps


def _nonexpert_match_rate(fa, fb):
    """fraction of A's NON-expert tensors whose value-fingerprint appears in B."""
    fb_avail = dict(fb)
    ne_total = ne_match = 0
    for fp, cnt in fa.items():
        numel = fp[0]
        is_expert = any(numel % e == 0 and numel >= e * 1024 for e in (256, 128, 64, 32))
        for _ in range(cnt):
            if is_expert:
                continue
            ne_total += 1
            if fb_avail.get(fp, 0) > 0:
                ne_match += 1
                fb_avail[fp] -= 1
    return ne_match, ne_total


def _scan(draft_dir, glob_pat, max_numel=2_000_000):
    """FAST source-trace: fingerprint ONLY the draft's small tensors once, then walk candidate
    train ckpts NEWEST→OLDEST, probing each on its small tensors. Stop at the first ≥95% match
    (that's the source). Big per-expert MoE blocks are never loaded (shape-peek skip)."""
    print(f">>> reference (converted draft) = {draft_dir}")
    fdraft = _small_fps(draft_dir, max_numel)
    print(f">>> draft small-tensor fingerprints (numel<= {max_numel:,}): {sum(fdraft.values())}\n")
    cands = [d for d in glob.glob(glob_pat) if os.path.isdir(d)]
    if not cands:
        sys.exit(f"no candidate dirs match glob: {glob_pat}")
    cands.sort(key=lambda d: os.path.getmtime(d), reverse=True)  # newest first
    print(">>> probing newest→oldest (small tensors only); stop at first ≥95%:")
    print("    match%   matched/total   mtime                dir")
    best = (0.0, 0, 0, None)
    import datetime as _dt
    for c in cands:
        try:
            fc = _small_fps(c, max_numel)
        except Exception as e:  # noqa: BLE001
            print(f"    (skip {c}: {e})")
            continue
        m, t = _nonexpert_match_rate(fc, fdraft)
        rate = m / t if t else 0.0
        ts = _dt.datetime.fromtimestamp(os.path.getmtime(c)).strftime("%Y-%m-%d %H:%M")
        hit = rate >= 0.95
        print(f"    {rate:6.1%}   {m:>5}/{t:<5}      {ts}    {c}{'  ← SOURCE' if hit else ''}")
        if rate > best[0]:
            best = (rate, m, t, c)
        if hit:
            print("\n" + "=" * 70)
            print(f"VERDICT: ✅ SOURCE FOUND → {c}")
            print("  ⇒ use THIS as the parity --ckpt (it IS the weights the ep4p5 serve ran).")
            sys.exit(0)
    print("\n" + "=" * 70)
    print(f"VERDICT: ❌ no train ckpt matches (best {best[0]:.1%} @ {best[3]}).")
    print("  ⇒ train-format source likely gone (mid-epoch dir overwritten after convert),")
    print("     or outside this glob / on another machine. Widen --scan or check 176.")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?", help="checkpoint dir A (e.g. the parity --ckpt, TRAIN format)")
    ap.add_argument("b", nargs="?", help="checkpoint dir B (e.g. the served CONVERTED draft dir)")
    ap.add_argument("--draft", help="SCAN MODE: the converted served draft to trace back to its train source")
    ap.add_argument("--scan", help="SCAN MODE: glob of candidate train ckpt dirs (e.g. "
                                   "'/home/a00652497/dspark_austin/run/ckpt_faithful_ep_*/[0-9]*')")
    args = ap.parse_args()

    if args.draft or args.scan:
        if not (args.draft and args.scan):
            sys.exit("scan mode needs BOTH --draft <converted dir> and --scan '<glob>'")
        _scan(args.draft, args.scan)
        return
    if not (args.a and args.b):
        sys.exit("pairwise mode needs two dirs: <A> <B>  (or use --draft/--scan)")

    print(f">>> A = {args.a}")
    print(f">>> B = {args.b}")
    fa, na, ea = _fingerprint_dir(args.a)
    fb, nb, eb = _fingerprint_dir(args.b)
    print(f">>> A: {na} tensors ({ea} expert-like) | B: {nb} tensors ({eb} expert-like)")

    # how many of A's tensors (by value fingerprint) are present in B
    matched = 0
    unmatched = []
    fb_avail = dict(fb)
    for fp, cnt in fa.items():
        take = min(cnt, fb_avail.get(fp, 0))
        matched += take
        fb_avail[fp] = fb_avail.get(fp, 0) - take
        if take < cnt:
            unmatched.append((fp, cnt - take))
    rate = matched / na if na else 0.0
    print(f"\n>>> {matched}/{na} of A's tensors have an identical-value match in B  "
          f"({rate:.1%})")
    # non-expert view: drop the big expert-count-leading tensors from the denominator
    a_nonexpert = sum(c for fp, c in fa.items())  # placeholder; recompute below
    # recompute A non-expert match rate directly
    ne_total = ne_match = 0
    fb_avail2 = dict(fb)
    for fp, cnt in fa.items():
        numel = fp[0]
        # treat as expert-like if numel is large AND divisible by a plausible expert count
        is_expert = any(numel % e == 0 and numel >= e * 1024 for e in (256, 128, 64, 32))
        for _ in range(cnt):
            if is_expert:
                continue
            ne_total += 1
            if fb_avail2.get(fp, 0) > 0:
                ne_match += 1
                fb_avail2[fp] -= 1
    if ne_total:
        print(f">>> non-expert tensors: {ne_match}/{ne_total} match "
              f"({ne_match / ne_total:.1%})  ← the decisive number")

    print("\n" + "=" * 70)
    if ne_total and ne_match / ne_total >= 0.95:
        print("VERDICT: ✅ SAME weights — A and B are the same trained state.")
        print("  ⇒ the parity divergence is NOT a ckpt mismatch; look elsewhere (forward path).")
    elif matched == 0:
        print("VERDICT: ❌ DIFFERENT weights — ZERO value matches.")
        print("  ⇒ A is NOT what the serve ran. Wrong epoch subdir or wrong run/convert source.")
    else:
        print("VERDICT: ⚠ PARTIAL — some tensors match, many don't.")
        print("  ⇒ likely a DIFFERENT epoch of the same run (shared frozen/verifier tensors match,")
        print("     trained tensors differ). Try the other epoch subdirs as A.")
    if unmatched[:5]:
        print("  sample A-tensors with no B match (numel, sum, abs.sum, absmax):")
        for fp, c in unmatched[:5]:
            print(f"    {fp}  x{c}")
    sys.exit(0)


if __name__ == "__main__":
    main()
