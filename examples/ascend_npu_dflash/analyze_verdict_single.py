#!/usr/bin/env python3
"""单份 verdict dump 的失效模式分析:草稿【断在哪里】,以及那些断点值不值得追。

和 ``analyze_verdict_dump.py`` 的分工:那个做【两份 run 的交叉表】(我们输、对方赢的清单),
必须有两份 dump;这个只吃一份,回答的是「这份草稿自己的失效长什么样」。

★ 核心是【首个拒绝点】,不是逐位接受率。逐位接受率是累积量,pos3 低既可能因为 pos3 本身难,
也可能只是因为 pos0-2 已经把大部分步筛掉了。把每一步的「断点落在第几个 slot」单独拎出来,
才看得到草稿是在哪一位失手的。

★ 断点按【目标自己的 top-1 概率】分档,这一步才让分析能推动决策:
    目标 p>0.9 还被我们草错  -> 目标很确定、草稿没跟上 = 草稿弱,训练买得到
    目标 p<0.5              -> 目标自己都不确定 = 谁都预测不了,追它是追噪声
  两者的下一步动作相反,不分开看就只能得出「再多训点」这种没有信息量的结论。

⚠ 每个请求末尾【整整一个 block】被剔除。一步最多发 K+1 个 token,生成在块中间停止时,越过
  停止点的 token 已经被采样器发出、记进 dump,之后才被调度器丢弃 —— 那些行必然是拒绝,且不
  代表草稿的真实失效(实测:released blk5 上正好多出 请求数 × block_size 行,全部为拒绝,
  剔掉之后接受率与引擎计数器分毫不差)。

USAGE
    python analyze_verdict_single.py --dir <verdict 目录> --tag released-blk5
    # 想看具体是哪些 token 断的,给分词器(在盒子上就是目标模型目录):
    python analyze_verdict_single.py --dir ... --tag ... --tokenizer /path/to/DeepSeek-V4-Flash-bf16
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# ⚠ 必须与 vllm_ascend/dspark_verdict_dumper.py 的 _REC 逐字段一致。定长二进制流没有自描述
# 头部,对不上不会报错,只会读出一堆看似合理的垃圾。
_REC = np.dtype([
    ("step", "<i4"), ("req", "<i4"), ("out_idx", "<i4"),
    ("draft_tok", "<i4"), ("target_tok", "<i4"), ("bonus_tok", "<i4"),
    ("slot", "i1"), ("accepted", "?"),
    ("target_top1_p", "<f2"), ("target_p_draft", "<f2"),
])

BANDS = ((0.9, 1.01, "目标很确定 p>0.9"),
         (0.5, 0.9, "中间 0.5-0.9"),
         (-0.01, 0.5, "目标也不确定 p<0.5"))


# 断点 token 的字符类别。★ 频次表只看得到最常断的那几个词,而长尾里同一类的几百个 token
# 各自频次都很低,逐个看永远发现不了模式;归到类别上一眼就出来。
FORMAT_KINDS = ("标点/符号", "换行", "空格")   # 排版类:定界符、空行节奏、分词边界


def make_kind(dec_raw):
    cache: dict[int, str] = {}

    def kind(tid: int) -> str:
        tid = int(tid)
        if tid in cache:
            return cache[tid]
        t = dec_raw(tid)
        core = t.strip()
        if t == "":
            k = "空/特殊"
        elif core == "":
            k = "换行" if "\n" in t else "空格"
        elif core.startswith("<") and core.endswith(">") or "｜" in core:
            k = "特殊标记"
        elif core.isdigit() or (core.lstrip("-").replace(".", "", 1).replace(",", "").isdigit()
                                and any(c.isdigit() for c in core)):
            k = "数字"
        elif all(not c.isalnum() for c in core):
            k = "标点/符号"
        elif any("\u4e00" <= c <= "\u9fff" for c in core):
            k = "中文"
        elif core.isalpha():
            k = "英文词/词片"
        else:
            k = "混合"
        cache[tid] = k
        return k

    return kind


def load(dirpath: str, tag: str) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(dirpath, f"verdict_{tag}_*.bin")))
    if not files:
        raise SystemExit(f"!! 没找到 verdict_{tag}_*.bin in {dirpath}")
    out, off = [], 0
    for f in files:
        a = np.fromfile(f, dtype=_REC)      # 末尾不足一条的残片自动丢弃
        rq = f.replace("verdict_", "reqs_")[: -len(".bin")] + ".txt"
        n_ids = sum(1 for _ in open(rq)) if os.path.exists(rq) else \
            (int(a["req"].max()) + 1 if len(a) else 0)
        # req 是每个 writer 进程各自从 0 编号的,不偏移就会把两个进程的请求 0 合成一条假流
        a = a.copy()
        a["req"] = a["req"] + off
        out.append(a)
        off += n_ids
    a = np.concatenate(out)
    print(f">>> {tag}: {len(files)} 个流, {len(a):,} 行, {len(np.unique(a['req'])):,} 个请求")
    return a


def steps(a: np.ndarray):
    """按 (req, step) 切块,返回每块的 [lo, hi) 以及排序后的数组。"""
    order = np.lexsort((a["slot"], a["step"], a["req"]))
    a = a[order]
    brk = np.flatnonzero((np.diff(a["req"]) != 0) | (np.diff(a["step"]) != 0)) + 1
    return a, np.r_[0, brk], np.r_[brk, len(a)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokenizer", help="目标模型目录;给了就把断点 token 解码成文本")
    ap.add_argument("--top", type=int, default=25, help="列出多少个最常见的断点 token")
    ap.add_argument("--samples", type=int, default=0,
                    help="每个置信档打印几个【带上下文的断点样例】(需要 --tokenizer)")
    ap.add_argument("--ctx", type=int, default=30, help="样例里回看多少个 token 作为上文")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = load(args.dir, args.tag)
    K = int(a["slot"].max()) + 1
    a, los, his = steps(a)

    # --- 每个请求剔掉最后一个 block(块越界,见模块注释)---
    last_step = {}
    for lo in los:
        last_step[int(a["req"][lo])] = max(last_step.get(int(a["req"][lo]), -1), int(a["step"][lo]))
    keep = np.array([not (int(a["step"][lo]) == last_step[int(a["req"][lo])]) for lo in los])
    print(f">>> block_size={K};剔掉每个请求末尾 1 个 block:"
          f"{(~keep).sum():,} / {len(los):,} 步(= 请求数),剩 {keep.sum():,} 步")

    los, his = los[keep], his[keep]
    n_steps = len(los)

    # --- 首个拒绝点 ---
    nacc = np.empty(n_steps, dtype=np.int32)     # 接受前缀长度 0..K
    brk_row = np.full(n_steps, -1, dtype=np.int64)   # 首个被拒 slot 的行号;全接受则 -1
    for i, (lo, hi) in enumerate(zip(los, his)):
        acc = a["accepted"][lo:hi]
        n = int(np.argmin(acc)) if not acc.all() else len(acc)
        nacc[i] = n
        if n < len(acc):
            brk_row[i] = lo + n

    print("\n" + "=" * 78)
    print("① 断在第几个 slot —— 每一步的接受前缀长度")
    print("=" * 78)
    print(f"{'前缀长度':>10} {'步数':>10} {'占比':>8}   含义")
    for n in range(K + 1):
        c = int((nacc == n).sum())
        what = "slot 0 就错(草稿第一个 token 就没跟上)" if n == 0 else \
               ("全接受(整块吃下)" if n == K else f"接受 {n} 个,断在 slot {n}")
        print(f"{n:>10} {c:>10,} {c/n_steps*100:>7.2f}%   {what}")

    # ★ 条件接受率 = P(第 s 位过 | 前 s 位都过)。逐位【累积】接受率会随位置单调下降,
    # 那主要是前缀在筛,不是后面的位置更难;条件量才是每一位自身的难度,形状也才有诊断价值:
    # 结构性 bug(mask / RoPE / block_size)的特征是条件量随位置急剧劣化,单纯的能力差是平移。
    print(f"\n{'slot':>10} {'到达步数':>10} {'过了':>10} {'条件接受率':>12}")
    for sl in range(K):
        reach = int((nacc >= sl).sum())
        passed = int((nacc > sl).sum())
        print(f"{sl:>10} {reach:>10,} {passed:>10,} {passed/max(reach,1)*100:>11.2f}%")
    print(f"\n   平均接受前缀 {nacc.mean():.3f}  ->  accept_len = {nacc.mean()+1:.3f}(+1 是 bonus token)")

    # --- 断点处目标有多确定 ---
    br = brk_row[brk_row >= 0]
    print("\n" + "=" * 78)
    print(f"② 断点处目标自己有多确定 —— 共 {len(br):,} 个断点")
    print("=" * 78)
    p1 = a["target_top1_p"][br].astype(np.float32)
    pd = a["target_p_draft"][br].astype(np.float32)
    print(f"{'档位':>22} {'断点数':>10} {'占比':>8} {'目标给草稿那个词的均值 p':>24}   结论")
    for lo_b, hi_b, name in BANDS:
        m = (p1 > lo_b) & (p1 <= hi_b)
        c = int(m.sum())
        verdict = {"目标很确定 p>0.9": "★ 草稿弱,训练买得到",
                   "中间 0.5-0.9": "灰区",
                   "目标也不确定 p<0.5": "噪声,追它没意义"}[name]
        print(f"{name:>22} {c:>10,} {c/max(len(br),1)*100:>7.2f}% {pd[m].mean() if c else 0:>24.4f}   {verdict}")

    # --- 前缀拖累:猜对了却因为前面已拒而不算 ---
    # ⚠ 必须只统计【保留的步】。越界块整块都是拒绝,混进来会同时抬高分母、压低接受率,
    # 得出的"前缀代价"就偏大 —— 而越界跟前缀规则毫无关系。
    kept_rows = np.zeros(len(a), dtype=bool)
    for lo, hi in zip(los, his):
        kept_rows[lo:hi] = True
    acc_all = a["accepted"][kept_rows]
    match = (a["draft_tok"][kept_rows] == a["target_tok"][kept_rows])
    wasted = int((match & ~acc_all).sum())
    print("\n" + "=" * 78)
    print("③ 被前缀拖累的 slot —— 草稿本来猜对了,但前面某位已经断了")
    print("=" * 78)
    print(f"   逐位猜对率 {match.mean()*100:.2f}%   实际接受率 {acc_all.mean()*100:.2f}%   "
          f"差额 {wasted:,} 行 = {wasted/len(acc_all)*100:.2f}%")
    print("   ⟹ 这部分是【前缀规则的代价】,不是草稿能力不足。把断点往后推一位,它们会自动变成接受。")

    # --- 断点上最常见的 token ---
    print("\n" + "=" * 78)
    print(f"④ 最常断在哪些 token 上(前 {args.top})")
    print("=" * 78)
    dt, tt = a["draft_tok"][br], a["target_tok"][br]
    pair = dt.astype(np.int64) * (1 << 21) + tt.astype(np.int64)
    uniq, cnt = np.unique(pair, return_counts=True)
    top = np.argsort(-cnt)[: args.top]
    dec = dec_raw = dec_raw_many = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
            tk = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            # 单 token 逐个 decode 会丢掉 SentencePiece 的前导空格语义,所以上文用整串 decode。
            dec_raw = lambda t: tk.decode([int(t)])            # noqa: E731
            dec_raw_many = lambda ts: tk.decode(list(ts))      # noqa: E731
            dec = lambda t: repr(dec_raw(t))                   # noqa: E731
        except Exception as e:  # noqa: BLE001
            print(f"   (分词器加载失败,只给 id:{type(e).__name__}: {e})")
    if dec is None:
        dec = lambda t: f"id={int(t)}"  # noqa: E731
    print(f"{'次数':>8} {'占断点':>8}   草稿给的 -> 目标要的")
    for i in top:
        d, t = int(uniq[i] >> 21), int(uniq[i] & ((1 << 21) - 1))
        print(f"{cnt[i]:>8,} {cnt[i]/len(br)*100:>7.2f}%   {dec(d):<28} -> {dec(t)}")

    # ---------------------------------------------------------------- ⑤ 字符类别
    # token 对的频次表看得到「最常断在哪几个词」,看不到「断在哪【类】东西上」。长尾里同一类
    # 的几百个不同 token 各自频次都很低,逐个看永远发现不了模式;归到类别上一眼就出来。
    if args.tokenizer and dec_raw is not None:
        print("\n" + "=" * 78)
        print("⑤ 断点 token 属于哪一类(草稿给的 / 目标要的)")
        print("=" * 78)

        kcached = make_kind(dec_raw)

        kd = [kcached(int(x)) for x in dt]
        kt = [kcached(int(x)) for x in tt]
        cats = sorted(set(kd) | set(kt))
        print(f"{'类别':>12} {'草稿给的':>12} {'目标要的':>12}   说明")
        note = {"数字": "算术/数值 —— 草稿没法凭语言模式猜出来的那类",
                "英文词/词片": "实词,语义预测失手",
                "标点/符号": "★排版:LaTeX 定界符 / 粗体标记 / 子句标点",
                "换行": "★排版:空行节奏、步骤边界",
                "空格": "★排版:分词边界",
                "中文": "中文实词", "混合": "", "空/特殊": "", "特殊标记": "EOS/BOS 等控制 token"}
        for c in cats:
            a_ = sum(1 for x in kd if x == c)
            b_ = sum(1 for x in kt if x == c)
            print(f"{c:>12} {a_:>11,} {b_:>11,}   {note.get(c,'')}")
        print("   ⟹ 两列差得多的类别 = 草稿【系统性地把 A 类词猜成 B 类词】,那是可命名的失效模式。")

    # ---------------------------------------------------------------- ⑥ 带上下文的样例
    if args.samples and args.tokenizer and dec_raw is not None:
        print("\n" + "=" * 78)
        print(f"⑥ 断点样例(每档 {args.samples} 个,上文 {args.ctx} 个 token)")
        print("=" * 78)
        # 重建每个请求实际发出的 token 序列。规则:该步接受的草稿 token,然后要么是断点处的
        # target_tok,要么(全接受时)是 bonus_tok。★ 没有 bonus_tok 这一列,全接受的步会
        # 静默少一个 token,上文就会错位 —— 这也是当初非加它不可的原因。
        stream: dict[int, dict[int, int]] = {}
        for lo, hi in zip(los, his):
            r = int(a["req"][lo])
            d = stream.setdefault(r, {})
            acc = a["accepted"][lo:hi]
            n = int(np.argmin(acc)) if not acc.all() else len(acc)
            for j in range(n):
                d[int(a["out_idx"][lo + j])] = int(a["draft_tok"][lo + j])
            if n < len(acc):
                d[int(a["out_idx"][lo + n])] = int(a["target_tok"][lo + n])
            else:
                d[int(a["out_idx"][lo + n - 1]) + 1] = int(a["bonus_tok"][lo])

        rng = np.random.default_rng(args.seed)
        for lo_b, hi_b, name in BANDS:
            sel = br[(p1 > lo_b) & (p1 <= hi_b)]
            if not len(sel):
                continue
            pick = rng.choice(sel, size=min(args.samples, len(sel)), replace=False)
            print(f"\n{'─' * 78}\n【{name}】 共 {len(sel):,} 个断点,抽 {len(pick)} 个\n{'─' * 78}")
            for row in pick:
                r, oi = int(a["req"][row]), int(a["out_idx"][row])
                d = stream.get(r, {})
                ctx_ids = [d[k] for k in range(max(0, oi - args.ctx), oi) if k in d]
                ctx = dec_raw_many(ctx_ids)
                print(f"  req {r}  step {int(a['step'][row])}  slot {int(a['slot'][row])}  "
                      f"out_idx {oi}   目标 top1 p={float(a['target_top1_p'][row]):.3f}  "
                      f"目标给草稿那词 p={float(a['target_p_draft'][row]):.3f}")
                print(f"    上文 …{ctx!r}")
                print(f"    草稿 -> {dec_raw(int(a['draft_tok'][row]))!r}"
                      f"      目标 -> {dec_raw(int(a['target_tok'][row]))!r}")

    # ---------------------------------------------------------------- ⑦ 类别 × 置信档
    # ⑤ 说「断在哪类词上」,② 说「目标当时多确定」。分开看各自都不足以决策:排版类断点如果
    # 都落在目标也不确定的档里,那是风格自由度、追不得;如果集中在 p>0.9,那是目标有确定
    # 写法而草稿没学会 —— 训练买得到。这张表把两者叉起来。
    if args.tokenizer and dec_raw is not None:
        kcat = make_kind(dec_raw)
        print("\n" + "=" * 78)
        print("⑦ 断点:目标要的那个 token 的类别 × 目标置信档")
        print("=" * 78)
        ktt = np.array([kcat(x) for x in tt])
        names = [n for _, _, n in BANDS]
        print(f"{'目标要的类别':>14} " + "".join(f"{n:>21}" for n in names) + f"{'合计':>10}")
        for c in sorted(set(ktt.tolist())):
            row_cnt, line = 0, f"{c:>14} "
            for (lo_b, hi_b, _n) in BANDS:
                v = int(((ktt == c) & (p1 > lo_b) & (p1 <= hi_b)).sum())
                row_cnt += v
                line += f"{v:>12,}({v / max(len(br), 1) * 100:>4.1f}%)"
            print(line + f"{row_cnt:>10,}")
        fmt_mask = np.isin(ktt, FORMAT_KINDS)
        print("-" * 78)
        line = f"{'★排版类占本档':>14} "
        for (lo_b, hi_b, _n) in BANDS:
            band = (p1 > lo_b) & (p1 <= hi_b)
            v = int((fmt_mask & band).sum())
            line += f"{v:>12,}({v / max(int(band.sum()), 1) * 100:>4.1f}%)"
        print(line + f"{int(fmt_mask.sum()):>10,}")
        print("   ★ 括号里是【该档之内】排版类的占比 —— 这一个数直接回答「训练重心要不要放到排版上」。")

    # ---------------------------------------------------------------- ⑧ 断点率 vs 位置
    # 猜想:排版约定(整篇用 display 还是 inline 数学、步骤间空几行)是长程的,由几百 token
    # 之前确立,而草稿的 sliding_window=128 —— 约定在窗口外,目标看得到草稿看不到。
    # 若成立,排版类断点的占比应当【随输出位置增加而上升】。
    # ⚠ 反证需要排除:DSpark 草稿吃目标第 40-42 层的 hidden state,长程信息本该从那里进来。
    # 所以这张表只能支持或证伪,不能单独定案。
    print("\n" + "=" * 78)
    print("⑧ 断点率随输出位置怎么变(测「排版约定跑出窗口」这个猜想)")
    print("=" * 78)
    step_oi = a["out_idx"][los]
    broke = brk_row >= 0
    edges = [0, 32, 64, 128, 192, 256, 384, 512, 768, 1 << 30]
    fmt_at_break = None
    if args.tokenizer and dec_raw is not None:
        kc8 = make_kind(dec_raw)
        fmt_at_break = np.zeros(len(los), dtype=bool)
        fmt_at_break[broke] = np.isin([kc8(x) for x in a["target_tok"][br]], FORMAT_KINDS)
    hdr = f"{'输出位置':>14} {'步数':>10} {'断点率':>9} {'平均前缀':>9}"
    if fmt_at_break is not None:
        hdr += f" {'断点中排版类占比':>18}"
    print(hdr)
    for i in range(len(edges) - 1):
        m = (step_oi >= edges[i]) & (step_oi < edges[i + 1])
        if not m.sum():
            continue
        hi_lab = edges[i + 1] if edges[i + 1] < (1 << 30) else "inf"
        line = (f"{f'{edges[i]}-{hi_lab}':>14} {int(m.sum()):>10,} "
                f"{broke[m].mean() * 100:>8.2f}% {nacc[m].mean():>9.3f}")
        if fmt_at_break is not None:
            nb = int((m & broke).sum())
            line += f" {(fmt_at_break[m & broke].mean() * 100 if nb else 0):>17.2f}%"
        print(line)
    print("   ★ 最后一列【单调上升】= 支持窗口猜想;基本持平 = 排版问题与位置无关,窗口不是主因。")

    # ---------------------------------------------------------------- ⑨ 控制 token 异常
    # 目标在流【中间】要 BOS 不正常。可能是请求边界泄漏(那说明 dump 还有一处没对齐,前面的
    # 结论都要打折),也可能只是收尾处的正常现象。看它们离请求末尾多远就能分辨。
    if args.tokenizer and dec_raw is not None:
        print("\n" + "=" * 78)
        print("⑨ 断点上的控制 token(EOS/BOS 之类)—— 真失效还是边界泄漏")
        print("=" * 78)
        kc9 = make_kind(dec_raw)
        spec = np.array([kc9(x) == "特殊标记" for x in tt])
        n_spec = int(spec.sum())
        if not n_spec:
            print("   没有。干净。")
        else:
            last_oi: dict[int, int] = {}
            for r_, oi_ in zip(a["req"], a["out_idx"]):
                r_ = int(r_)
                if oi_ > last_oi.get(r_, -1):
                    last_oi[r_] = int(oi_)
            d2end = np.array([last_oi[int(a["req"][row])] - int(a["out_idx"][row])
                              for row in br[spec]])
            print(f"   {n_spec:,} 个断点的【目标】是控制 token,占断点 {n_spec / len(br) * 100:.2f}%")
            print(f"   距本请求输出末尾的距离:中位数 {np.median(d2end):.0f}"
                  f"   ≤5 的占 {(d2end <= 5).mean() * 100:.1f}%"
                  f"   ≤20 的占 {(d2end <= 20).mean() * 100:.1f}%")
            print("   ⟹ 绝大多数贴着末尾 = 收尾处的正常现象(草稿没押准何时停),可忽略;")
            print("      散布在流中间 = 【请求边界泄漏】,dump 还有一处没对齐,前面的结论要打折。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
