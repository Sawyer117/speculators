#!/usr/bin/env python3
"""`main_proj` 怎么加权那三层 aux hidden state —— released 和我们自己的草稿对比。

WHY
---
DSpark 把目标模型 3 层 aux hidden state 拼起来过一个线性层再喂草稿:

    data.py:103   torch.cat(data["hidden_states"][:-1], dim=-1)   # 3 × 4096 = 12288
    fc / mtp.0.main_proj.weight                                    # [4096, 12288]

`[:-1]` 只取 3 个 aux(config 里的 target_layer_ids = 40/41/42),**final 那层不在里面**
——它是目标侧。所以「最后一层权重更大吗」在这个矩阵里问的是 **layer 42 那个列块**。

列方向 12288 = 3 个 4096 的块,块 b 只乘 layer (40+b) 的 hidden state。所以
「这一层有多重要」就是块 b 的贡献占比。

⚠ 权重范数不等于贡献
--------------------
transformer 里 hidden state 的 RMS 随深度增长(残差一路累加)。42 层激活大,学出来的
权重反而会小去补偿。所以只看 ‖W_b‖ 会系统性低估深层。给了 --hs 时本脚本直接算
**‖W_b · h_b‖ 在真实位置上的均值**,那才是贡献;没给就只报权重范数,并在表头写明。

⚠ 这个探针能回答什么、不能回答什么
----------------------------------
能:released 和我们的草稿,对三层的加权方式一不一样。差很多 = 我们学到的是另一种组合,
    这是训练质量的信号。
不能:证明「推理抖动导致深层更重要」。那个假说有符号歧义 —— 噪声随层累积的信噪比论证
    预测深层权重【更小】,而「深层离 logits 近、信息更多」预测【更大】,两者方向相反,
    且后者不需要任何抖动就成立(它是零假设)。所以无论量出哪个方向都不构成证据。
    要测抖动本身,应该同一条语料 dump 两次 HS,比较【每一层】的 |Δ| 是否随层增长。

USAGE
-----
  python aux_mix_probe.py /path/to/released_draft_bf16_standalone \
                          /path/to/dsv4_dspark_blk15_ep5p0_vllm-77w
  # 带上一份 HS dump,算真实贡献而不只是权重范数:
  python aux_mix_probe.py <ckpt...> --hs /path/to/hs_000123.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 纯离线读权重,不需要 NPU。不关掉的话 import torch 会自动拉 torch_npu,在没 source CANN
# 的 shell 里直接报 "Failed to load the backend extension"。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

NAME_HINTS = ("main_proj", "fc.weight", "fc_main_proj")


def _find_files(d: str) -> list[str]:
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(idx):
        m = json.load(open(idx, encoding="utf-8"))["weight_map"]
        return sorted({os.path.join(d, v) for v in m.values()})
    import glob  # noqa: PLC0415
    return sorted(glob.glob(os.path.join(d, "*.safetensors")))


# safetensors 的 numpy 后端【不支持 BF16】—— 而 released_draft_bf16_standalone 正是 bf16,
# 用 numpy 打开会直接抛。有 torch 就用 torch 读(它什么 dtype 都认),再转 float32。
def _reader():
    try:
        import torch  # noqa: PLC0415
        from safetensors import safe_open  # noqa: PLC0415

        def _get(fh, k):
            return torch.Tensor.float(fh.get_tensor(k)).cpu().numpy()
        return "pt", safe_open, _get
    except Exception:  # noqa: BLE001
        from safetensors import safe_open  # noqa: PLC0415

        def _get(fh, k):
            return fh.get_tensor(k)
        return "numpy", safe_open, _get


def load_main_proj(d: str):
    """返回 (name, W float32 [out, 3*H], 来源文件)。fp8 块量化会按 scale 反量化。

    找不到时会说清楚是【目录里没有权重文件】还是【有文件但没有形状对的键】,并把看到的
    候选键打出来 —— 第一版只说「找不到」,而实际原因是目录名打错了,查了半天。
    """
    import numpy as np  # noqa: PLC0415
    fw, safe_open, _get = _reader()

    files = _find_files(d)
    if not os.path.isdir(d):
        print(f"!! {d}: 目录不存在")
        return None, None, None
    if not files:
        print(f"!! {d}: 目录里没有 .safetensors —— 里面有:"
              f"{sorted(os.listdir(d))[:8]}")
        return None, None, None
    seen: list[str] = []

    for f in files:
        try:
            with safe_open(f, framework=fw) as fh:
                keys = list(fh.keys())
                cands = [k for k in keys
                         if any(h in k for h in NAME_HINTS) and k.endswith("weight")]
                for k in cands:
                    sl = fh.get_slice(k)
                    shp = list(sl.get_shape())
                    seen.append(f"{k}{shp}")
                    if len(shp) != 2:
                        continue
                    # [H, 3H] 是常规存法;有的导出会转置成 [3H, H],两种都收。
                    transposed = shp[0] == 3 * shp[1]
                    if not (shp[1] == 3 * shp[0] or transposed):
                        continue
                    w = _get(fh, k).astype(np.float32)
                    if transposed:
                        w = w.T.copy()
                    skey = k[: -len("weight")] + "scale"
                    if skey in keys:                  # fp8 块量化
                        s = _get(fh, skey).astype(np.float32)
                        br = -(-w.shape[0] // s.shape[0])
                        bc = -(-w.shape[1] // s.shape[1])
                        big = np.repeat(np.repeat(s, br, axis=0), bc, axis=1)
                        w = w * big[: w.shape[0], : w.shape[1]]
                        note = f"fp8×scale{list(s.shape)} 块 {br}×{bc}"
                    else:
                        note = str(sl.get_dtype())
                    return k, w, f"{os.path.basename(f)}  ({note})"
        except Exception as e:  # noqa: BLE001
            print(f"    (跳过 {os.path.basename(f)}: {type(e).__name__}: "
                  f"{' '.join(str(e).split())[:110]})", file=sys.stderr)
    if seen:
        print(f"!! {d}: 有 main_proj/fc 键但形状都不是 [H, 3H]:{seen[:6]}")
    else:
        print(f"!! {d}: {len(files)} 个 safetensors 里一个 main_proj/fc 权重都没有")
    return None, None, None


def hs_rms(path: str, n_aux: int):
    """从一份 HS dump 里取每个 aux 层的 RMS。dump = [seq, 3 aux + 1, H]。"""
    import numpy as np  # noqa: PLC0415
    fw, safe_open, _get = _reader()
    with safe_open(path, framework=fw) as fh:
        for k in fh.keys():
            t = _get(fh, k)
            if t.ndim == 3 and t.shape[1] >= n_aux:
                t = t.astype(np.float32)
                return [float(np.sqrt((t[:, b, :] ** 2).mean())) for b in range(n_aux)], k, t
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", nargs="+", help="草稿权重目录(released 放第一个便于对照)")
    ap.add_argument("--hs", default="", help="一份 HS dump(.safetensors),用来算真实贡献")
    ap.add_argument("--layers", default="40,41,42", help="仅用于打标签")
    args = ap.parse_args()

    import numpy as np  # noqa: PLC0415

    labels = [s.strip() for s in args.layers.split(",")]
    rms = hs_arr = None
    if args.hs:
        rms, key, hs_arr = hs_rms(args.hs, len(labels))
        if rms:
            print(f"HS dump {os.path.basename(args.hs)}  张量 {key}  "
                  f"形状 {list(hs_arr.shape)}")
            print("  每层 RMS: " + "  ".join(f"L{l}={r:.4f}" for l, r in zip(labels, rms)))
        else:
            print(f"⚠ {args.hs} 里找不到 [seq, ≥{len(labels)}, H] 的张量,退回只报权重范数")
        print()

    rows = []
    for d in args.ckpt:
        name, w, src = load_main_proj(d)
        if w is None:
            continue
        h = w.shape[0]
        nb = w.shape[1] // h
        blocks = [w[:, b * h:(b + 1) * h] for b in range(nb)]
        fro = [float(np.linalg.norm(b)) for b in blocks]
        mabs = [float(np.abs(b).mean()) for b in blocks]
        contrib = None
        if hs_arr is not None and hs_arr.shape[2] == h:
            # 真实贡献:‖W_b · h_b‖ 在位置上的均值。比 ‖W_b‖ 诚实得多。
            take = min(512, hs_arr.shape[0])
            contrib = []
            for b in range(nb):
                hb = hs_arr[:take, b, :].astype(np.float32)      # [T, H]
                y = hb @ blocks[b].T                             # [T, out]
                contrib.append(float(np.linalg.norm(y, axis=1).mean()))
        rows.append((os.path.basename(d.rstrip("/")), name, src, h, nb, fro, mabs, contrib))

    if not rows:
        return 2

    print("=" * 96)
    print("  main_proj 三个列块的占比(块 b 只乘 layer %s 的 hidden state)" % "/".join(labels))
    print("=" * 96)
    for nm, key, src, h, nb, fro, mabs, contrib in rows:
        tot = sum(fro) or 1.0
        print(f"\n▌{nm}")
        print(f"   {key}  [{h}, {h * nb}]   {src}")
        print("   " + " " * 12 + "".join(f"{('L' + labels[b]):>14}" for b in range(nb)))
        print("   ‖W_b‖_F    " + "".join(f"{fro[b]:>14.2f}" for b in range(nb)))
        print("   占比        " + "".join(f"{100 * fro[b] / tot:>13.2f}%" for b in range(nb)))
        print("   mean|w|     " + "".join(f"{mabs[b]:>14.3e}" for b in range(nb)))
        if contrib:
            ct = sum(contrib) or 1.0
            print("   ★‖W_b·h_b‖ " + "".join(f"{contrib[b]:>14.2f}" for b in range(nb)))
            print("   ★贡献占比   " + "".join(f"{100 * contrib[b] / ct:>13.2f}%" for b in range(nb)))
    print()
    if len(rows) > 1:
        base = rows[0]
        # ★ 2026-09-23:第一版这张差值表用的是权重范数,而上面表头报的是【贡献占比】。
        #   两个口径混在一起看会得出相反的结论 —— 有 --hs 时一律用贡献。
        use_c = base[7] is not None and all(r[7] is not None for r in rows[1:])
        bvec = base[7] if use_c else base[5]
        bt = sum(bvec) or 1.0
        print("=" * 96)
        print(f"  相对 {base[0]} 的{'贡献' if use_c else '权重范数'}占比差(正 = 我们更看重这一层)")
        if not use_c:
            print("  ⚠ 没有 --hs,这里比的是权重范数,不是贡献。深层激活 RMS 更大 ⟹ 权重偏小,")
            print("    只看范数会系统性低估深层。")
        print("=" * 96)
        print("   " + " " * 30 + "".join(f"{('L' + labels[b]):>14}" for b in range(base[4])))
        for nm, _k, _s, _h, nb, fro, _m, contrib in rows[1:]:
            vec = contrib if use_c else fro
            t = sum(vec) or 1.0
            print(f"   {nm[:28]:<30}" + "".join(
                f"{100 * vec[b] / t - 100 * bvec[b] / bt:>+13.2f}pt" for b in range(nb)))
    print()
    print("读法 —— 这个表能说和不能说的:")
    print("  能说:released 和我们的加权方式差多少。差很多 = 我们学到的是另一种组合方式。")
    print("  不能说:『抖动让深层更重要』。噪声随层累积的信噪比论证预测深层权重【更小】,")
    print("         『深层离 logits 近』预测【更大】,方向相反且后者是零假设。")
    print("  要测抖动本身:同一条语料 dump 两次 HS,比较每一层的 |Δ| 是否随层增长。")
    if not args.hs:
        print("  ⚠ 只有权重范数。深层激活 RMS 更大会让它的权重偏小 —— 带 --hs 才是真实贡献。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
