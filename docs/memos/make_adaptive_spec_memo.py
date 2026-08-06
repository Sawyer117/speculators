#!/usr/bin/env python3
"""Build the technical memo deck: DSpark adaptive speculation × graph mode on Ascend NPU.

Findings are code-grounded (vllm-ascend @ dspark-parity, vLLM PR #48692, Ascend op docs).
Run:  python3 make_memo_ppt.py --out <file.pptx>
"""
from __future__ import annotations

import argparse

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ── palette ────────────────────────────────────────────────────────────────────
INK = RGBColor(0x1B, 0x22, 0x2E)      # near-black body text
MUTE = RGBColor(0x5B, 0x66, 0x77)     # secondary text
ACCENT = RGBColor(0x1F, 0x6F, 0xEB)   # headings / rules
GOOD = RGBColor(0x0E, 0x7A, 0x4B)     # "this is fine" green
WARN = RGBColor(0xB4, 0x53, 0x09)     # "this is the cost" amber
BG = RGBColor(0xFF, 0xFF, 0xFF)
BAND = RGBColor(0xF2, 0xF5, 0xF9)     # table header / callout fill

W, H = Inches(13.333), Inches(7.5)    # 16:9


def _txbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def _p(tf, text, *, size=16, bold=False, color=INK, space_before=6, space_after=0,
       level=0, align=PP_ALIGN.LEFT, first=False, italic=False):
    para = tf.paragraphs[0] if first else tf.add_paragraph()
    para.alignment = align
    para.level = level
    para.space_before = Pt(space_before)
    para.space_after = Pt(space_after)
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return para


def _slide(prs, title, kicker=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])          # blank
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    tf = _txbox(s, Inches(0.62), Inches(0.42), Inches(12.1), Inches(1.0))
    _p(tf, title, size=28, bold=True, color=INK, first=True, space_before=0)
    if kicker:
        _p(tf, kicker, size=13.5, color=MUTE, space_before=4)
    # accent rule
    line = s.shapes.add_shape(1, Inches(0.62), Inches(1.42), Inches(1.5), Emu(26000))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()
    line.shadow.inherit = False
    return s


def _table(slide, rows, x, y, w, col_w=None, font=12.5, header=True, row_h=0.34):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, x, y, w, Inches(row_h * nr)).table
    if col_w:
        total = sum(col_w)
        for i, frac in enumerate(col_w):
            shp.columns[i].width = Emu(int(w * frac / total))
    shp.first_row = header
    for r, row in enumerate(rows):
        shp.rows[r].height = Inches(row_h)
        for c, val in enumerate(row):
            cell = shp.cell(r, c)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = BAND if (header and r == 0) else BG
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            para = cell.text_frame.paragraphs[0]
            cell.text_frame.word_wrap = True
            run = para.add_run()
            run.text = str(val)
            run.font.size = Pt(font)
            run.font.bold = header and r == 0
            run.font.color.rgb = INK if (header and r == 0) else INK
    return shp


def _code(slide, lines, x, y, w, h, font=12):
    box = slide.shapes.add_shape(1, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = BAND
    box.line.color.rgb = RGBColor(0xD8, 0xDF, 0xE8)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left, tf.margin_top = Inches(0.16), Inches(0.10)
    for i, ln in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_before, para.space_after = Pt(1), Pt(1)
        run = para.add_run()
        run.text = ln
        run.font.size = Pt(font)
        run.font.name = "Consolas"
        run.font.color.rgb = INK
    return box


def build(out: str) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ── 1. title ───────────────────────────────────────────────────────────────
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    tf = _txbox(s, Inches(0.9), Inches(2.25), Inches(11.5), Inches(3.0))
    _p(tf, "Adaptive speculative decoding under graph mode", size=40, bold=True, first=True, space_before=0)
    _p(tf, "What it costs on Ascend NPU, and where the real blockers are", size=21, color=MUTE, space_before=10)
    _p(tf, "Technical memo · findings grounded in vllm-ascend source, vLLM PR #48692 and the Ascend operator docs",
       size=13, color=MUTE, space_before=22)

    # ── 2. the question ────────────────────────────────────────────────────────
    s = _slide(prs, "The question", "Why “graph-mode efficiency” is the crux of adaptive speculation")
    tf = _txbox(s, Inches(0.62), Inches(1.75), Inches(12.1), Inches(4.6))
    _p(tf, "Adaptive speculation gives each request its own verification length K_i, so the target model only "
           "verifies tokens worth verifying. That saves compute in theory.", size=17, first=True, space_before=0)
    _p(tf, "Device graphs want the opposite: fixed shapes, fixed addresses, a stable execution path.", size=17, space_before=12)
    _p(tf, "So the question is not “does adaptive speculation reduce target tokens” — it does. It is:", size=17, space_before=12)
    _p(tf, "Do the graph updates, workspace queries, tiling, host metadata and padding needed to make per-round "
           "K_i fit a device graph cost more than the compute they save?", size=18, bold=True, color=ACCENT, space_before=10)
    _p(tf, "This memo answers three sub-questions from the source: (1) what PR #48692 actually delivers, "
           "(2) whether device-side query lengths are possible on Ascend, (3) what remains to be adapted.",
       size=14, color=MUTE, space_before=18)

    # ── 3. PR #48692 — what it really is ───────────────────────────────────────
    s = _slide(prs, "Finding 1 — PR #48692 is narrower than assumed", "vLLM main · [MRV2][Spec Decode] Adaptive Speculative Decoding · closed, not merged")
    _table(s, [
        ["", "Common assumption", "What the PR states"],
        ["Verification budget", "Allocated online from confidence head", "Explicitly NOT implemented — a user-provided\n num_speculative_tokens_per_batch_size is used"],
        ["Speedup", "Demonstrated", "“no significant speedup is measurable;\n that is not the objective”"],
        ["Backends", "General", "FLASH_ATTENTION only"],
        ["Status", "In flight", "Closed, needs-rebase · 28 files, +1240/−110"],
    ], Inches(0.62), Inches(1.8), Inches(12.1), col_w=[0.17, 0.31, 0.52], row_h=0.62)
    tf = _txbox(s, Inches(0.62), Inches(5.3), Inches(12.1), Inches(1.6))
    _p(tf, "What it does deliver: variable-length per-request speculation that stays FULL-CUDA-Graph compatible — "
           "attention-metadata flags for variable-length decode, structured outputs, logprobs, mixed prefill/decode.",
       size=15, first=True, space_before=0)
    _p(tf, "Measured benefit is acceptance rate, not latency: adaptive at an effective K=5 budget reaches 3.71 "
           "against 3.45 for static K=5 and 3.92 for static K=7 (SPEED-Bench average).", size=15, space_before=8)
    _p(tf, "⟹ Treat it as a reference for “how variable length coexists with a full graph”, not as evidence of speedup.",
       size=15, bold=True, color=ACCENT, space_before=10)

    # ── 4. Finding 2 — device-side lengths ─────────────────────────────────────
    s = _slide(prs, "Finding 2 — device-side query lengths already exist on Ascend",
               "…but only on one of the two operator families")
    _table(s, [
        ["Operator family", "Sequence-length contract", "Evidence"],
        ["FIA v2\nnpu_fused_infer_attention_score_v2", "Host-side int array (SymInt[])", "Ascend op docs; every vllm-ascend call site\n builds it with .tolist()"],
        ["DSA / lightning indexer / SFA\ntorch.ops._C_ascend.npu_vllm_*", "torch.Tensor (device)", "execute_sparse_flash_attention_process(\n   actual_seq_lengths_query: torch.Tensor, …)\n call sites pass query_start_loc[1:].clone()"],
    ], Inches(0.62), Inches(1.8), Inches(12.1), col_w=[0.28, 0.24, 0.48], row_h=0.95)
    tf = _txbox(s, Inches(0.62), Inches(5.0), Inches(12.1), Inches(1.9))
    _p(tf, "DeepSeek-V4-Flash runs on the DSA path, not the FIA path.", size=17, bold=True, color=GOOD, first=True, space_before=0)
    _p(tf, "So for this model the hardest prerequisite — keeping K_i on the device instead of round-tripping through "
           "a Python list — is already satisfied by the operators it uses. The host-list constraint applies to the "
           "generic FIA path used by dense models, which is the same scope limit PR #48692 has on the GPU side "
           "(FLASH_ATTENTION only).", size=15, space_before=8)

    # ── 5. Finding 3 — workspace cache key ─────────────────────────────────────
    s = _slide(prs, "Finding 3 — the workspace cache is keyed by exactly what adaptive K changes",
               "vllm_ascend/attention/attention_v1.py")
    _code(s, [
        "num_tokens = attn_metadata.actual_seq_lengths_q[-1]   # total tokens this round",
        "workspace  = graph_params.workspaces.get(num_tokens)  # cache keyed by that total",
        "...",
        "elif workspace is None:                               # query only on a miss",
        "    workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(...)",
    ], Inches(0.62), Inches(1.85), Inches(12.1), Inches(1.75))
    tf = _txbox(s, Inches(0.62), Inches(3.85), Inches(12.1), Inches(3.0))
    _p(tf, "Static K: the total token count is constant every round → cache hits → no cost.",
       size=16, color=GOOD, first=True, space_before=0)
    _p(tf, "Adaptive K_i: the total changes every round → a miss every round → _get_max_workspace is called "
           "per layer, per step.", size=16, color=WARN, space_before=8)
    _p(tf, "A second path is more expensive still: when _use_max_workspace_for_fia_graph is set, the query runs "
           "unconditionally on every call so the largest workspace across layer variants can be kept.",
       size=15, space_before=8)
    _p(tf, "Mitigation is structural, not a micro-optimisation: key the cache by a bucket "
           "(batch bucket × total-verify-token bucket) rather than the exact token count, and pre-allocate at capture time.",
       size=16, bold=True, color=ACCENT, space_before=12)

    # ── 6. cost ledger ─────────────────────────────────────────────────────────
    s = _slide(prs, "Where the extra cost actually comes from", "Per decode step, relative to a static-K full graph")
    _table(s, [
        ["Source", "Static K", "Adaptive K_i", "Comment"],
        ["FIA workspace query", "cached", "miss per round, per layer", "cache key = total token count"],
        ["Graph task update", "per layer", "per layer", "graph_task_update_begin / …_end already exists"],
        ["Host metadata", "small", "grows with ragged batch", "FIA path builds lengths on host"],
        ["Padding waste", "none", "depends on bucket width", "the price of a reusable graph"],
        ["Target compute saved", "—", "the entire point", "only meaningful when verify is expensive"],
    ], Inches(0.62), Inches(1.8), Inches(12.1), col_w=[0.24, 0.14, 0.26, 0.36], row_h=0.46)
    tf = _txbox(s, Inches(0.62), Inches(5.15), Inches(12.1), Inches(1.8))
    _p(tf, "The last row decides the whole question. On an 8B model the PR author could not measure a speedup because "
           "verification is already cheap. The saving scales with how expensive the target forward is — which is why a "
           "large MoE target is the case where this is worth doing at all.", size=15, first=True, space_before=0)

    # ── 7. what to verify next ─────────────────────────────────────────────────
    s = _slide(prs, "What to measure before building anything", "Cheap experiments that decide whether to proceed")
    tf = _txbox(s, Inches(0.62), Inches(1.8), Inches(12.1), Inches(5.0))
    for i, (head, body) in enumerate([
        ("1 · Size the prize",
         "Measure the target-side token count under static K versus an adaptive allocation on the same traffic. "
         "If the reduction is small, no amount of graph engineering pays for itself."),
        ("2 · Four-way comparison",
         "static-K full graph · adaptive-K eager · adaptive-K piecewise graph · adaptive-K full graph. "
         "Report verified tokens, graph replay hit rate, padding tokens, host metadata time, workspace/tiling time, "
         "graph-update time, attention time, end-to-end decode latency."),
        ("3 · Audit the K_i path for host round-trips",
         "Confirm that confidence-head output, prefix selection, index compaction, position and slot mapping all stay "
         "on device — no .cpu(), .tolist() or implicit D2H between the confidence head and attention."),
        ("4 · Decide the graph key",
         "Bucket by (batch size, total verify tokens) rather than capturing one graph per distinct K_i vector, "
         "and pre-allocate workspace per bucket at capture time."),
    ]):
        _p(tf, head, size=17, bold=True, color=ACCENT, first=(i == 0), space_before=0 if i == 0 else 16)
        _p(tf, body, size=14.5, space_before=4)

    # ── 8. bottom line ─────────────────────────────────────────────────────────
    s = _slide(prs, "Bottom line")
    tf = _txbox(s, Inches(0.62), Inches(1.85), Inches(12.1), Inches(5.0))
    _p(tf, "“NPU is less graph-friendly for adaptive speculation” is true, but not for the reason usually given.",
       size=19, bold=True, first=True, space_before=0)
    _p(tf, "It is not that the hardware cannot do dynamic shapes. On the DSA path that DeepSeek-V4-Flash uses, "
           "sequence lengths are already device tensors — the prerequisite everyone worries about is met.",
       size=16, space_before=14)
    _p(tf, "The genuine costs are narrower and fixable: a workspace cache keyed by the exact token count, "
           "per-layer graph-task updates, and host-side metadata on the generic FIA path.",
       size=16, space_before=10)
    _p(tf, "And the GPU-side reference does not yet prove a speedup — it was measured on a model where verification "
           "is cheap. The case for doing this at all rests on an expensive target model, which is exactly the case "
           "worth measuring first.", size=16, space_before=10)
    _p(tf, "Recommendation: quantify the saving before engineering the graph path.",
       size=18, bold=True, color=ACCENT, space_before=18)

    # ── 9. sources ─────────────────────────────────────────────────────────────
    s = _slide(prs, "Sources")
    tf = _txbox(s, Inches(0.62), Inches(1.85), Inches(12.1), Inches(5.0))
    for i, ln in enumerate([
        "vLLM PR #48692 — [MRV2][Spec Decode] Adaptive Speculative Decoding, Initial Support (closed, needs-rebase)",
        "vllm-ascend — vllm_ascend/attention/attention_v1.py (FIA v2 workspace cache, graph task update)",
        "vllm-ascend — vllm_ascend/attention/dsa_v1.py (DSA / lightning indexer call sites)",
        "vllm-ascend — vllm_ascend/device/device_op.py (execute_sparse_flash_attention_process signature)",
        "vllm-ascend — vllm_ascend/attention/utils.py (cache_graph_workspace)",
        "vllm-ascend — vllm_ascend/worker/v2/model_runner.py (MRV2 support is present)",
        "Ascend operator documentation — torch_npu.npu_fused_infer_attention_score / npu_fusion_attention",
    ]):
        _p(tf, "· " + ln, size=14, first=(i == 0), space_before=0 if i == 0 else 9)

    prs.save(out)
    print(f"saved: {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="adaptive-spec-graph-mode-ascend.pptx")
    build(ap.parse_args().out)
