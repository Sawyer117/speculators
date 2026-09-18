#!/usr/bin/env python3
"""把一整个训练 run 的 checkpoint 批量导出成 vLLM 能加载的草稿目录。

WHY
---
"导出来 eval" 一直是三条手敲命令 × 每个 epoch:``--inspect`` / ``convert`` /
``verify``,路径靠记、epoch 编号靠数、``--config-from`` 靠背。每多一个 epoch 就多三次
出错机会,而错了不会报错 —— 会安静地服务一个配置错的草稿。这个脚本把整条链子收成一条
命令:自己找 run、自己给 ckpt 标 epoch、自己拼 config、转完自己验。

⚠ 默认【只打印计划,什么都不写】。确认无误后再加 ``--go``。

它替你挡掉的四个坑
------------------
1. **``--config-from`` 把 γ 抄错。** 转换器是整份拷贝 released 草稿的 config.json,而那份
   写死 ``dspark_block_size=5``。block-16 训出来的草稿(γ=15)照抄就会带着 5 出厂。
   本脚本改成 **released 打底 + 用 ckpt 自己的 config 覆盖**,并把每一处不同都打出来。
   (当前 pin ``4ce367a`` 上 ``dspark_block_size`` 只在 ``deepseek_v4/dspark.py:126``
   赋给一个没人再读的字段,实际 γ 由 ``NUM_SPEC`` 决定 —— 所以这条今天不致命,但
   出厂目录自己描述错了自己,下一个 pin 读它就是事故。``sliding_window`` 同理,而那条
   serve 是真读的。)
2. **epoch 编号。** trainer 存在整数目录 ``0/ 1/ 2/…`` 里,``epochE_end`` /
   ``epochE_step<S>`` 只是软链;而且 **E 是 0-indexed**:``epoch4_end`` = 训了 **5.0** 个
   epoch。目录名和"训了几个 epoch"差一,历史上已经数错过。这里统一按训练量命名
   (``ep5p0`` = 5.0 epoch),和 ``eval_blk15_drafts.sh`` 的清单对得上。
3. **转到一半的 ckpt。** 存一个 ckpt 要 ~3 分钟。落在写入过程中的目录会转出一份缺张量的
   权重,而 safetensors 读得下去 —— 只在 serve 时表现为精度莫名其妙。凡是 10 分钟内动过的
   目录一律跳过(``--force`` 可越过)。
4. **出厂目录别的账号读不了。** 转换器已经 chmod 0755/0644,这里再核一遍并打出来。

USAGE
-----
    # 1) 看计划(不写任何东西)。不给 --run 就挑最新的那个 run。
    python examples/ascend_npu_dflash/export_run_ckpts.py
    python examples/ascend_npu_dflash/export_run_ckpts.py --run 20260916_031128

    # 2) 真的转(转完自动 verify,逐个打 bit-exact 计数)
    python examples/ascend_npu_dflash/export_run_ckpts.py --run 20260916_031128 --go

    # 只转某几个整数 ckpt 目录 / 换输出根 / 重转已存在的
    ... --only 4,3  |  --out-root /home/canada_group_folder/ckpt  |  --force

纯 CPU(torch + safetensors),和训练/serve 抢不到资源,可以在 serve 起着的时候跑。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 训练 run 目录的候选。launch_a3_dsv4_dspark.sh 用 RUN=$HOME/dsv4_run;A2 的 austin 树在
# dspark_austin/run 下。183 上登录账号是 n84449292 而开发树在 a00652497 —— $HOME 和仓库
# 不在一起,所以两个都得找。
RUN_ROOTS = (
    "$HOME/dsv4_run",
    "$HOME/dspark_austin/run",
    "/home/n84449292/dsv4_run",
    "/home/a00652497/dsv4_run",
    "/home/a00652497/dspark_austin/run",
)

# 权重根。A3-176 = /home/...(草稿直接放根下);A2 = /share/...(草稿在 dsv4_dspark_drafts/);
# A3 共享 NFS = /mnt/nfs/...。顺序即优先级。
CKPT_ROOTS = (
    "/home/canada_group_folder/ckpt",
    "/share/canada_group_folder/ckpt",
    "/mnt/nfs/canada_group_folder/ckpt",
)

RELEASED = "released_draft_bf16_standalone"

# ckpt 里的"朴素名" -> serve config 里的 dspark_* 名。值的变换写在 _merge_config 里,
# 因为 γ 那条不是改名是减一。
_SERVE_FIELDS = (
    # (serve key,                     ckpt key(s) 候选,                      变换)
    ("dspark_block_size",             ("block_size",),                        "gamma"),
    ("dspark_noise_token_id",         ("mask_token_id", "noise_token_id"),    None),
    ("dspark_target_layer_ids",       ("aux_hidden_state_layer_ids",
                                       "target_layer_ids"),                   "list"),
    ("dspark_markov_rank",            ("markov_rank",),                       None),
    ("sliding_window",                ("window_size",),                       None),
)
# ⚠ num_nextn_predict_layers(草稿几层)故意【不】从 config 里查:ckpt 的 config 里
# `num_hidden_layers` 至少出现两次 —— 草稿深度在 transformer_layer_config 里是 3,而某些
# 结构里还挂着 target 的 61。摊平后谁赢取决于 dict 顺序,赢错了 serve 就去建 61 层草稿。
# 层数在【权重里】是白纸黑字的,直接数 layers.{n}. 即可。


def _flatten(obj, out: dict, prefix: str = "") -> dict:
    """把嵌套 config 摊平成 {leaf_key: value}。

    ckpt 的 config.json 是 speculators 的 pydantic 结构:形状字段在
    ``transformer_layer_config`` 里,方法字段在顶层。按叶子名查比按路径查稳 —— 结构换了
    (历史上换过)按名字仍然找得到。同名冲突时【浅的赢】,顶层才是方法配置。
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and not _is_scalar_list(v):
                _flatten(v, out, f"{prefix}{k}.")
            elif k not in out:
                out[k] = v
    return out


def tensor_names(d: Path) -> list[str]:
    """读 safetensors 的头,拿全部张量名 —— 不 import torch,也不把权重读进内存。

    格式:前 8 字节 = 头长(小端 u64),接着就是那么长的一段 JSON。
    """
    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        return list(json.loads(idx.read_text())["weight_map"])
    f = d / "model.safetensors"
    if not f.is_file():
        return []
    with f.open("rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        head = json.loads(fh.read(n))
    return [k for k in head if k != "__metadata__"]


def n_draft_layers(names: list[str]) -> int | None:
    ns = {int(m.group(1)) for k in names for m in [re.match(r"layers\.(\d+)\.", k)] if m}
    return max(ns) + 1 if ns else None


def _is_scalar_list(v) -> bool:
    return isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v)


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


def find_runs(explicit_root: str | None) -> list[Path]:
    """所有 ckpt_faithful_* 目录,按 mtime 新到旧。"""
    roots = [_expand(explicit_root)] if explicit_root else [_expand(r) for r in RUN_ROOTS]
    runs: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        if not r.is_dir():
            continue
        for d in sorted(r.glob("ckpt_*")):
            rp = str(d.resolve())
            if d.is_dir() and rp not in seen:
                seen.add(rp)
                runs.append(d)
    return sorted(runs, key=lambda d: d.stat().st_mtime, reverse=True)


def find_ckpt_root(explicit: str | None) -> Path | None:
    """放 released 草稿的那个权重根 —— 转换输出也放这里(serve 直接能读)。"""
    cands = [_expand(explicit)] if explicit else [_expand(c) for c in CKPT_ROOTS]
    for c in cands:
        if (c / RELEASED / "config.json").is_file():
            return c
    for c in cands:                      # 没有 released 也行,至少目录在
        if c.is_dir():
            return c
    return None


def label_ckpts(run: Path) -> list[dict]:
    """列出 run 下的整数 ckpt 目录,给每个标上"训了多少 epoch"。

    ⚠ 两套编号必须掰清楚:
      * 目录名 ``0/ 1/ 2/…`` = **上一个完成的 epoch 索引**(0-indexed)。
      * 软链 ``epochE_end`` = epoch 索引 E 结束 = 训了 **E+1** 个 epoch。
      * 软链 ``epochE_step<S>`` = epoch 索引 E 的中点 = 训了 **E+0.5** 个。
      * 同一个整数目录会被"中点存"和"末尾存"先后写两次(末尾覆盖中点),所以一个目录上
        可能挂着两个软链 —— 此时 **以 _end 为准**,里面躺的是末尾那份。
    软链缺失(被清过)时回落到 training_state.json:local_step==0 ⟹ 是末尾存。
    """
    # 软链 -> 它指向的整数目录
    links: dict[str, list[str]] = {}
    for p in run.iterdir():
        m = re.fullmatch(r"epoch(\d+)_(end|step\d+)", p.name)
        if m and p.is_symlink():
            links.setdefault(os.path.basename(os.path.realpath(p)).strip("/"), []).append(p.name)

    out: list[dict] = []
    for p in sorted(run.iterdir(), key=lambda x: (len(x.name), x.name)):
        if p.is_symlink() or not p.is_dir() or not re.fullmatch(r"\d+", p.name):
            continue
        st_path = p / "training_state.json"
        st = json.loads(st_path.read_text()) if st_path.is_file() else {}
        names = links.get(p.name, [])
        end = [n for n in names if n.endswith("_end")]
        mid = [n for n in names if "_step" in n]

        if end:
            epochs = int(re.match(r"epoch(\d+)_", end[0]).group(1)) + 1.0
            via = end[0]
        elif mid:
            epochs = int(re.match(r"epoch(\d+)_", mid[0]).group(1)) + 0.5
            via = mid[0]
        elif st:
            # local_step 0 = 刚跑完一个 epoch 存的;否则是中点存
            epochs = st.get("epoch", int(p.name)) + (1.0 if not st.get("local_step") else 0.5)
            via = "training_state.json"
        else:
            epochs = int(p.name) + 1.0
            via = "目录名(无软链无 state,按 end 猜)"

        has_w = (p / "model.safetensors").is_file() or (p / "model.safetensors.index.json").is_file()
        out.append({
            "dir": p,
            "epochs": epochs,
            "label": f"ep{int(epochs)}p{int(round((epochs % 1) * 10))}",
            "via": via,
            "global_step": st.get("global_step"),
            "mtime": p.stat().st_mtime,
            "weights": has_w,
            "links": names,
        })
    return out


def _merge_config(released: dict, ck_flat: dict,
                  n_layers: int | None) -> tuple[dict, list[str], list[str]]:
    """released config 打底,用 ckpt 自己的值覆盖。返回 (config, 改动行, 警告行)。"""
    cfg = dict(released)
    changes: list[str] = []
    warns: list[str] = []
    for serve_key, ck_keys, xform in _SERVE_FIELDS:
        src = next((k for k in ck_keys if k in ck_flat), None)
        if src is None:
            warns.append(f"ckpt config 里找不到 {'/'.join(ck_keys)} → {serve_key} 沿用 released 的 "
                         f"{released.get(serve_key)!r}")
            continue
        val = ck_flat[src]
        if xform == "gamma":
            # ckpt 的 block_size 是【块宽】= anchor + γ 个 mask 槽;serve 的
            # dspark_block_size 是 γ。差的那 1 就是 anchor(slot 0,训练时 loss 被 mask)。
            val = int(val) - 1
        elif xform == "list":
            val = list(val)
        old = cfg.get(serve_key)
        cfg[serve_key] = val
        if old != val:
            changes.append(f"{serve_key}: {old!r} → {val!r}   (取自 ckpt 的 {src}"
                           f"{'−1=γ' if xform == 'gamma' else ''})")
    if n_layers:
        old = cfg.get("num_nextn_predict_layers")
        cfg["num_nextn_predict_layers"] = n_layers
        if old != n_layers:
            changes.append(f"num_nextn_predict_layers: {old!r} → {n_layers}   "
                           f"(数权重里的 layers.{{n}}.,不是查 config)")
    else:
        warns.append("权重里数不出 layers.{n}. —— num_nextn_predict_layers 沿用 released 的 "
                     f"{cfg.get('num_nextn_predict_layers')!r}")
    # serve 走 EAGLE3 的 aux 通路读这个名字;为空会回落到 eagle3 默认的 4 层,
    # 和草稿 main_proj 要的 3H 对不上,第一次提议就崩。
    cfg["eagle_aux_hidden_state_layer_ids"] = list(cfg["dspark_target_layer_ids"])
    return cfg, changes, warns


def _run(cmd: list[str]) -> int:
    print("    $ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run 的时间戳片段(如 20260916_031128);省略=最新的那个")
    ap.add_argument("--run-root", help="run 目录的父目录(省略=在已知的几个位置里找)")
    ap.add_argument("--out-root", help="草稿输出根(省略=放 released 草稿的那个权重根)")
    ap.add_argument("--only", help="只处理这些整数 ckpt 目录,逗号分隔(如 4,3)")
    ap.add_argument("--prefix", default="dsv4_dspark", help="输出目录名前缀")
    ap.add_argument("--suffix", default="vllm-77w", help="输出目录名后缀")
    ap.add_argument("--go", action="store_true", help="真的转换(默认只打印计划)")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出 / 不跳过刚写过的 ckpt")
    ap.add_argument("--fresh-sec", type=int, default=600,
                    help="多少秒内动过的 ckpt 视为「可能还在写」而跳过(默认 600)")
    args = ap.parse_args()

    runs = find_runs(args.run_root)
    if not runs:
        print(f"!! 在这些位置都没找到 ckpt_* 目录:{', '.join(RUN_ROOTS)}\n"
              f"   用 --run-root <dir> 指一下。", file=sys.stderr)
        return 2
    if args.run:
        picked = [r for r in runs if args.run in r.name]
        if not picked:
            print(f"!! 没有名字里带 {args.run!r} 的 run。现有:", file=sys.stderr)
            for r in runs:
                print(f"     {r}", file=sys.stderr)
            return 2
        run = picked[0]
    else:
        # 挑最新的,但【必须挑一个真的存过权重的】。一个刚建好还没存过 ckpt 的 run,或者被
        # 清空过的旧 run,mtime 可能比谁都新 —— 默认落到它身上,输出就是一句"没有 ckpt",
        # 而真正要导的那个 run 就在下一行。
        run = next((r for r in runs if any(e["weights"] for e in label_ckpts(r))), runs[0])
        if run is not runs[0]:
            print(f">>> 最新的 {runs[0].name} 里没有带权重的 ckpt,跳过它,选 {run.name}")

    ckpt_root = find_ckpt_root(args.out_root)
    if ckpt_root is None:
        print(f"!! 找不到权重根(要有 {RELEASED}/config.json):{', '.join(CKPT_ROOTS)}\n"
              f"   用 --out-root <dir> 指一下。", file=sys.stderr)
        return 2
    # A2 把草稿收在 dsv4_dspark_drafts/ 下;A3-176 直接摊在根里。已有哪种就跟哪种。
    out_root = ckpt_root / "dsv4_dspark_drafts" if (ckpt_root / "dsv4_dspark_drafts").is_dir() \
        else ckpt_root
    rel_cfg_path = ckpt_root / RELEASED / "config.json"

    entries = label_ckpts(run)
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        entries = [e for e in entries if e["dir"].name in want]

    print("=" * 96)
    print(f"run          : {run}")
    print(f"权重根       : {ckpt_root}")
    print(f"输出根       : {out_root}")
    print(f"released cfg : {rel_cfg_path}{'' if rel_cfg_path.is_file() else '   !! 不存在'}")
    print(f"其它 run     : " + (", ".join(r.name for r in runs if r != run) or "(无)"))
    print("=" * 96)
    if not entries:
        print("!! 这个 run 下没有整数 ckpt 目录 —— 一次都还没存过?")
        return 2
    if not rel_cfg_path.is_file():
        print("!! 没有 released 草稿的 config.json,拼不出 serve 配置。--out-root 指到有它的那个根。",
              file=sys.stderr)
        return 2

    released = json.loads(rel_cfg_path.read_text())

    # ── ckpt 的 config:同一个 run 里每个 epoch 都一样,取第一个有的即可 ──────────────
    ck_cfg_path = next((e["dir"] / "config.json" for e in entries
                        if (e["dir"] / "config.json").is_file()), None)
    if ck_cfg_path is None:
        print("!! ckpt 目录里没有 config.json —— 无法核对 γ / sliding_window。", file=sys.stderr)
        return 2
    ck_flat = _flatten(json.loads(ck_cfg_path.read_text()), {})
    cfg, changes, warns = _merge_config(released, ck_flat,
                                        n_draft_layers(tensor_names(ck_cfg_path.parent)))
    gamma = int(cfg["dspark_block_size"])

    print(f"\nckpt config  : {ck_cfg_path}")
    print(f"γ(每步草稿几个 token) = {gamma}   ⟹  serve 必须 NUM_SPEC={gamma}"
          f"(块注意力非因果,低于它不报错只出错数)")
    if changes:
        print("released config 被 ckpt 覆盖的字段:")
        for c in changes:
            print(f"    {c}")
    else:
        print("released config 与 ckpt 完全一致(没有需要覆盖的字段)")
    for w in warns:
        print(f"    ⚠ {w}")

    # ── 逐个 ckpt 的计划 ────────────────────────────────────────────────────────────
    now = time.time()
    plan: list[tuple[dict, Path]] = []
    print(f"\n{'ckpt':>5}  {'训练量':>7}  {'global_step':>11}  {'来源':<22}  输出")
    print("-" * 96)
    for e in entries:
        name = f"{args.prefix}_blk{gamma}_{e['label']}_{args.suffix}"
        dst = out_root / name
        note = ""
        skip = False
        if not e["weights"]:
            note, skip = "!! 没有 model.safetensors —— 跳过", True
        elif now - e["mtime"] < args.fresh_sec and not args.force:
            note, skip = (f"!! {int(now - e['mtime'])}s 前刚动过,可能还在写 —— 跳过"
                          f"(--force 越过)"), True
        elif (dst / "model.safetensors").is_file() and not args.force:
            note, skip = "已存在 —— 跳过(--force 重转)", True
        print(f"{e['dir'].name:>5}  {e['epochs']:>6.1f}ep  {str(e['global_step'] or '?'):>11}  "
              f"{e['via']:<22}  {name}")
        if note:
            print(f"{'':>5}  {note}")
        if not skip:
            plan.append((e, dst))

    if not plan:
        print("\n没有要转的 —— 上面每一条都被跳过了。")
    elif not args.go:
        print(f"\n>>> 计划转 {len(plan)} 个。什么都还没写。确认后加 --go 重跑同一条命令。")

    if not args.go or not plan:
        # 干跑时把【全部】ckpt 都列进 eval 清单(包括本次跳过的:多半是上一轮已经转好的),
        # 这样这条命令直接可用,不用再手补。
        shown = plan or [(e, out_root / f"{args.prefix}_blk{gamma}_{e['label']}_{args.suffix}")
                         for e in entries if e["weights"]]
        _print_eval_hint(gamma, [f"{e['label']}-blk{gamma}|{d.name}" for e, d in shown])
        return 0

    # ── 真的转 ─────────────────────────────────────────────────────────────────────
    # 合并后的 config 落一份临时文件给转换器当 --config-from。转换器对这几个字段有硬检查
    # (缺任何一个就退出),所以这里出错会当场炸,不会安静地服务错配置。
    tmp_cfg = Path(os.environ.get("TMPDIR", "/tmp")) / f"dspark_serve_config_blk{gamma}.json"
    tmp_cfg.write_text(json.dumps(cfg, indent=2))
    print(f"\n>>> serve config 写到 {tmp_cfg}(转换器的 --config-from)")

    conv = REPO_ROOT / "scripts" / "convert_dspark_to_vllm.py"
    veri = REPO_ROOT / "scripts" / "verify_dspark_conversion.py"
    ok, bad = [], []
    for i, (e, dst) in enumerate(plan, 1):
        print("\n" + "=" * 96)
        print(f"### [{i}/{len(plan)}] {e['dir']}  ({e['epochs']:.1f}ep)  →  {dst}")
        print("=" * 96, flush=True)
        if dst.exists() and args.force:
            print(f"    --force:先删掉已有的 {dst}")
            shutil.rmtree(dst)
        rc = _run([sys.executable, str(conv), "--in", str(e["dir"]), "--out", str(dst),
                   "--config-from", str(tmp_cfg)])
        if rc != 0:
            print(f"!! 转换失败 rc={rc}")
            bad.append((e["label"], "convert"))
            continue
        rc = _run([sys.executable, str(veri), "--in", str(e["dir"]), "--out", str(dst)])
        if rc != 0:
            print(f"!! 校验失败 rc={rc} —— 这份【不要用】")
            bad.append((e["label"], "verify"))
            continue
        ok.append((e["label"], dst))

    print("\n" + "=" * 96)
    print(f"### 导出完成:成功 {len(ok)}  失败 {len(bad)}")
    for lbl, d in ok:
        print(f"    ✓ {lbl:<8} {d}")
    for lbl, stage in bad:
        print(f"    ✗ {lbl:<8} 卡在 {stage}")
    print("=" * 96)
    _print_eval_hint(gamma, [f"{lbl}-blk{gamma}|{d.name}" for lbl, d in ok])
    return 1 if bad else 0


def _print_eval_hint(gamma: int, entries: list[str]) -> None:
    if not entries:
        return
    print("\n>>> 跑 eval(在 serve 那台机上,eval_blk15_drafts.sh 会自己起/停 serve):")
    print(f"    ENTRIES_OVERRIDE='{' '.join(entries)}' \\")
    print(f"    NUM_SPEC={gamma} DATASET=gsm8k \\")
    print("      nohup bash examples/ascend_npu_dflash/eval_blk15_drafts.sh "
          "> ~/eval_blk_driver.log 2>&1 &")


if __name__ == "__main__":
    sys.exit(main())
