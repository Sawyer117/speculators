#!/usr/bin/env python3
"""HS 值不对时,先别猜 —— 把「哪个切片 / 要不要过 norm / 有没有错位」穷举成一张表。

WHY
---
2026-09-20:`dsv4_hs_integrity_check.py` MODE 1 在新栈的 dump 上报 **79.9% mismatch**,
而且最后一个十分位仍有 80%(本该 ~0%)。这个数太大了,大到不像「数据被写坏」——
拿**中间层**过 lm_head 的典型 top-1 命中率就在 20% 上下,正好对应 80% mismatch。

所以在下任何结论之前,要先把这三件事分开:

  1. **取错切片**    dump 是 [seq, L_aux+1, H];约定是 [:, -1] = 最终层。万一反了呢?
  2. **少过一层 norm** lm_head 吃的是 final RMSNorm **之后**的张量。如果 dump 的是
                      norm 之前的残差流,argmax 会大面积错,但数据本身是好的 ——
                      补一次 norm 就能用,不用重 dump 几 TB。
  3. **错位**        argmax(h_i) 该对齐 token_ids[i+1]。差一格就全错。

一次把 L×{raw, normed}×{shift −1,0,+1} 全跑出来。**只要有一格接近 0%,数据就是好的**,
差的只是读法;**全都很高**,才轮到怀疑 dump 本身,那时再上 MODE 2(独立 HF 前向)。

注意:prompt 段是用户给的 token,本来就预测不准;只有 response 段才该 ~0%。Arrow 里的
`loss_mask` 标的就是 response,但 dump 文件里没有它,所以这里用**位置尾部**近似
(``--tail-frac``,默认只统计后 40% 的位置),并且同时打印全量数,两个都给你看。

USAGE
-----
    python hs_value_probe.py --hs-dir ~/dsa_fault_ab/dumps_A \
        --model-dir /home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16
    ... --files 3 --max-pos 512      # 更快
纯 CPU,只加载 lm_head(~1.8 GB)+ model.norm.weight(16 KB)。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_HEAD_KEYS = ("lm_head.weight", "head.weight")
_NORM_KEYS = ("model.norm.weight", "norm.weight", "model.final_layernorm.weight")


def _find_tensor(model_dir: str, candidates: tuple[str, ...], what: str):
    from safetensors import safe_open  # noqa: PLC0415

    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        wmap = json.load(open(idx))["weight_map"]
        for k in candidates:
            if k in wmap:
                with safe_open(os.path.join(model_dir, wmap[k]), framework="pt") as fh:
                    return k, fh.get_tensor(k)
    for p in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(p, framework="pt") as fh:
            ks = set(fh.keys())
            for k in candidates:
                if k in ks:
                    return k, fh.get_tensor(k)
    raise SystemExit(f"!! 在 {model_dir} 里找不到 {what}(试过 {candidates})")


def rms_norm(x, w, eps: float = 1e-6):
    import torch  # noqa: PLC0415

    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w.float()


def mismatch(h, W, ids, shift: int, chunk: int, topk: int = 0) -> tuple[float, int, float]:
    """argmax(h_i @ W.T) 与 ids[i+shift] 的不一致率;topk>0 时另给 top-k 命中率。

    ★ 为什么要 top-k:argmax 不中【不等于】值是垃圾。目标 token 稳定落在 top-10 里
      = 方向对、只是数值有偏差(精度/非确定性);连 top-10 都进不去 = 真的不相干。
      这两种要的下一步完全不同,而它们的 argmax mismatch 长得一模一样。
    """
    T = h.shape[0]
    lo = max(0, -shift)
    hi = min(T, T - shift)
    if hi - lo <= 0:
        return float("nan"), 0, float("nan")
    bad = tot = hit = 0
    for s in range(lo, hi, chunk):
        e = min(s + chunk, hi)
        logits = h[s:e].float() @ W.T
        tgt = ids[s + shift: e + shift]
        bad += int((logits.argmax(-1) != tgt).sum())
        if topk:
            tk = logits.topk(topk, dim=-1).indices
            hit += int((tk == tgt.unsqueeze(-1)).any(-1).sum())
        tot += e - s
    return 100.0 * bad / max(tot, 1), tot, (100.0 * hit / max(tot, 1) if topk else float("nan"))


def resp_mismatch(h, W, ids, shift: int, chunk: int, mask, ) -> tuple[float, int]:
    """只在 mask 为真的【目标位置】上算 mismatch —— 即 ids[i+shift] 属于 response 段。"""
    T = h.shape[0]
    lo, hi = max(0, -shift), min(T, T - shift)
    bad = tot = 0
    for s in range(lo, hi, chunk):
        e = min(s + chunk, hi)
        m = mask[s + shift: e + shift]
        if not bool(m.any()):
            continue
        logits = h[s:e][m].float() @ W.T
        tgt = ids[s + shift: e + shift][m]
        bad += int((logits.argmax(-1) != tgt).sum())
        tot += int(m.sum())
    return (100.0 * bad / max(tot, 1)), tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hs-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--files", type=int, default=3, help="取几个文件(默认 3,够分辨了)")
    ap.add_argument("--max-pos", type=int, default=768, help="每个文件最多算几个位置")
    ap.add_argument("--tail-frac", type=float, default=0.4, help="尾部这一段近似 response 段")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--topk", type=int, default=10, help="另报「目标 token 在 top-k 内」的比例")
    # ★ 决定性的两条:拿 Arrow 原始行来对。
    #   (1) dump 里的 token_ids 和 Arrow 的 input_ids 逐个比 —— 不一致 = dumper 把 id 写错了
    #       /错位了,那是 dumper 的 bug,和模型无关;
    #   (2) 用 loss_mask 只统计 **response** 段。「尾部 40%」只是近似,prompt 长的行会把
    #       response 段稀释掉,得出的高 mismatch 没有意义。response 段才是该 ~0% 的地方。
    ap.add_argument("--arrow", help="训练 Arrow 目录;给了就做 token_ids 对拍 + loss_mask 限定")
    ap.add_argument("--id-offset", type=int, default=0,
                    help="Arrow 行号 = 文件 id − 这个值。生产约定 id==row 所以是 0;"
                         "今晚那批 pilot 用了 --id-base 768 --start-row 0,所以填 768")
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    ds = None
    if args.arrow:
        from datasets import load_from_disk  # noqa: PLC0415

        ds = load_from_disk(args.arrow)
        if hasattr(ds, "keys") and not hasattr(ds, "num_rows"):
            ds = ds[next(iter(ds.keys()))]
        try:
            ds = ds.with_format(None)
        except Exception:  # noqa: BLE001
            pass
        print(f"Arrow: {args.arrow}  {len(ds)} 行  列={ds.column_names}  "
              f"(行号 = 文件 id − {args.id_offset})")

    fs = sorted(glob.glob(f"{args.hs_dir}/hs_*.safetensors"))
    if not fs:
        print(f"!! {args.hs_dir} 里没有 hs_*.safetensors", file=sys.stderr)
        return 2
    fs = fs[: args.files]

    hk, W = _find_tensor(args.model_dir, _HEAD_KEYS, "lm_head")
    nk, NW = _find_tensor(args.model_dir, _NORM_KEYS, "final norm weight")
    # ★ ckpt 权重是 bf16,而我们按 float32 算 —— 不统一会直接
    #   `expected m1 and m2 to have the same dtype`。载入时 cast 一次,别在内层循环里转。
    W = W.float()
    NW = NW.float()
    print(f"lm_head='{hk}' {tuple(W.shape)}   final_norm='{nk}' {tuple(NW.shape)}")
    print(f"样本 {len(fs)} 个文件,每个最多 {args.max_pos} 个位置\n")

    # ── 先看幅度:post-norm 的张量和残差流的量级差很多 ─────────────────────────
    d0 = load_file(fs[0])
    hs0 = d0["hidden_states"]
    L = hs0.shape[1]
    print(f"格式:hidden_states {tuple(hs0.shape)}  token_ids {tuple(d0['token_ids'].shape)}")
    print("每个切片的 RMS(post-final-norm 的那一片,量级应当和 norm.weight 同量级,"
          "且明显不同于中间层残差):")
    for s in range(L):
        v = hs0[:, s, :].float()
        print(f"   slice[{s}{' = 最后一片' if s == L - 1 else ''}]  "
              f"rms={v.pow(2).mean().sqrt():.4f}  absmax={v.abs().max():.2f}")
    print(f"   norm.weight            rms={NW.float().pow(2).mean().sqrt():.4f}\n")

    # ── 穷举:切片 × {raw, normed} × shift ────────────────────────────────────
    combos = [(s, n, k) for s in range(L) for n in (False, True) for k in (1, 0, -1)]
    acc: dict[tuple, list] = {c: [[0, 0], [0, 0]] for c in combos}   # [全量, 尾部] 各 [bad, tot]
    hits: dict[tuple, float] = {}

    resp_acc: dict[tuple, list] = {c: [0.0, 0] for c in combos}
    ids_checked = ids_equal = 0

    for f in fs:
        d = load_file(f)
        hs, ids = d["hidden_states"], d["token_ids"].long()
        lm = None
        if ds is not None:
            row = int(os.path.basename(f)[3:].split(".")[0]) - args.id_offset
            if 0 <= row < len(ds):
                a_ids = torch.tensor(ds[row]["input_ids"], dtype=torch.long)
                ids_checked += 1
                n = min(len(a_ids), len(ids))
                same = bool(len(a_ids) == len(ids)) and bool((a_ids == ids).all())
                ids_equal += int(same)
                if not same:
                    diff = int((a_ids[:n] != ids[:n]).sum())
                    print(f"  ⚠ {os.path.basename(f)} 的 token_ids 和 Arrow 行 {row} 不一致:"
                          f"长度 {len(ids)} vs {len(a_ids)},前 {n} 位里 {diff} 处不同")
                if "loss_mask" in ds.column_names:
                    lm = torch.tensor(ds[row]["loss_mask"], dtype=torch.bool)
        T = min(hs.shape[0], args.max_pos)
        hs, ids = hs[:T], ids[:T]
        if lm is not None:
            lm = lm[:T]
        t0 = int(T * (1.0 - args.tail_frac))
        for (s, n, k) in combos:
            h = hs[:, s, :]
            if n:
                h = rms_norm(h, NW)
            r_all, n_all, _ = mismatch(h, W, ids, k, args.chunk)
            r_tl, n_tl, hit = mismatch(h[t0:], W, ids[t0:], k, args.chunk, topk=args.topk)
            if n_all:
                acc[(s, n, k)][0][0] += r_all * n_all / 100.0
                acc[(s, n, k)][0][1] += n_all
            if n_tl:
                acc[(s, n, k)][1][0] += r_tl * n_tl / 100.0
                acc[(s, n, k)][1][1] += n_tl
                hits[(s, n, k)] = hits.get((s, n, k), 0.0) + hit * n_tl / 100.0
            # response 段(loss_mask==1):这才是「该 ~0%」的地方
            if lm is not None and bool(lm.any()):
                pred_ok = resp_mismatch(h, W, ids, k, args.chunk, lm)
                if pred_ok[1]:
                    resp_acc[(s, n, k)][0] += pred_ok[0] * pred_ok[1] / 100.0
                    resp_acc[(s, n, k)][1] += pred_ok[1]

    print(f"{'切片':<8}{'norm':<7}{'shift':<7}{'全量 mismatch':>14}{'尾部 mismatch':>14}"
          f"{f'尾部 top{args.topk} 命中':>16}{'response 段':>14}")
    print("-" * 68)
    best = None
    for (s, n, k) in combos:
        (ba, ta), (bt, tt) = acc[(s, n, k)]
        ra = 100.0 * ba / max(ta, 1)
        rt = 100.0 * bt / max(tt, 1)
        tag = f"[{s}]" + ("=末" if s == L - 1 else "")
        hk_ = 100.0 * hits.get((s, n, k), 0.0) / max(tt, 1)
        rr = resp_acc[(s, n, k)]
        rs = (f"{100.0 * rr[0] / rr[1]:>12.2f}%" if rr[1] else f"{'—':>13}")
        print(f"{tag:<8}{'是' if n else '否':<6}{k:>4}   {ra:>12.2f}% {rt:>13.2f}% {hk_:>14.2f}% {rs}")
        if best is None or rt < best[0]:
            best = (rt, s, n, k, ra)
    print("-" * 68)
    if ids_checked:
        print(f"\n★ token_ids 对拍:{ids_equal}/{ids_checked} 个文件与 Arrow 完全一致"
              + ("  → dumper 的 id 没写错,错位/串行都排除" if ids_equal == ids_checked
                 else "  → ⚠ dumper 把 token_ids 写错了,这是 dumper 的 bug,先修它"))
        rbest = min(((resp_acc[c][0] / resp_acc[c][1] * 100.0, c)
                     for c in combos if resp_acc[c][1]), default=None)
        if rbest:
            rv, (cs, cn, ck) = rbest
            print(f"★ response 段(loss_mask==1)最好的一格:切片[{cs}] "
                  f"{'过' if cn else '不过'} norm shift={ck} → {rv:.2f}%")
            if rv < 5:
                print("  ✅ response 段几乎全中 ⟹ **dump 是好的**,之前的 79.9% 是因为把 prompt 段"
                      "(用户给的 token,本来就预测不了)算进去了。可以放行批量 dump。")
            elif rv < 30:
                print("  ⚠ response 段也有明显偏差 —— 可能这批语料不是本模型贪心生成的。"
                      "先确认 Arrow 的来源(rollout 回流 vs 原始 SFT 语料)。")
            else:
                print("  ❌ response 段也很差 ⟹ 要么语料不是本模型生成的,要么 dump 真有问题。"
                      "下一步:自产自验 —— 用这台 serve 贪心生成一段,再把 prompt+生成 回灌做"
                      "prefill dump,那段的 mismatch 必须 ~0%。")

    rt, s, n, k, ra = best
    print(f"\n最好的一格:切片[{s}] {'过 norm' if n else '不过 norm'} shift={k} "
          f"→ 尾部 {rt:.2f}%(全量 {ra:.2f}%)")
    print()
    if rt < 5:
        print("✅ 数据是好的,只是读法不对。按上面这一格改 dumper/检查器的约定即可,")
        print("   **不需要重新 dump**。")
        if n:
            print("   ⟹ dump 的是 final RMSNorm【之前】的张量。要么 dumper 里补一次 norm,")
            print("      要么训练侧读的时候补 —— 但两边必须一致,而且要和旧语料的约定对齐。")
        if k != 1:
            print(f"   ⟹ 还存在 shift={k} 的错位,dumper 的 token_ids 和 hidden_states 没对齐。")
    elif rt < 20:
        print("⚠️ 最好的一格也有几个到十几个百分点 —— 像是【部分】损坏(过订阅/并发写坏),")
        print("   而不是读法问题。对比 conc=1 的 dump:run_hs_consistency_check.sh。")
    elif max(hits.get(c, 0.0) / max(acc[c][1][1], 1) * 100.0 for c in combos) > 60:
        print("⚠️ argmax 都不中,但目标 token 大比例落在 top-k 里 ⟹ 方向对、数值有偏差,")
        print("   不是「捕获了完全不相干的东西」。优先怀疑:基准本身(rollout 不是这套栈/不是贪心)、")
        print("   bf16 非确定性、或者 prompt 段占比过高。先用 Arrow 的 loss_mask 只统计 response 段。")
    else:
        print("❌ 所有读法都很差,连 top-k 都不中 ⟹ dump 出来的值本身就不对。")
        print("   下一步 MODE 2(独立 HF 前向)定位是捕获点错了还是数值被写坏:")
        print("   dsv4_hs_integrity_check.py --hf-model <dir>  (重,需要整模型)")
        print("   在那之前【绝对不要】开始批量 dump。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
