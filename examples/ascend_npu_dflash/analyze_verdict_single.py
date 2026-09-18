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

# 英文虚词/功能词。⑫ 用它把「英文词/词片」拆成承载意义的实词和不承载的虚词 —— 草稿把 the
# 猜成 a,和把 32 算成 24,是完全不同的病,而 ⑤/⑦ 的字符形态分类看不出这个区别。
# ⚠ 这是一份【人工清单】,边界必然粗糙(比如 "each"/"per" 在数学题里其实承载意义)。
# 它只用于分诊、给人看,不参与任何数值结论。
FUNC_WORDS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "he", "she", "they", "we",
    "you", "his", "her", "their", "our", "your", "i", "them", "him", "us", "me",
    "is", "are", "was", "were", "be", "been", "being", "am", "s", "re", "ve", "ll", "d", "m", "t",
    "has", "have", "had", "do", "does", "did", "will", "would", "can", "could", "should", "may",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "as", "into", "over", "under",
    "and", "or", "but", "so", "if", "then", "than", "because", "since", "while", "when", "where",
    "which", "who", "what", "how", "not", "no", "there", "here", "now", "also", "thus", "hence",
    "each", "per", "let", "us", "both", "all", "any", "some", "more", "most", "less", "such",
}


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

    # top-k 侧流(可选)。⚠ 必须【按主流的文件顺序逐个配对】再拼接,不能各自 glob 再 concat:
    # 两个前缀的排序未必给出同样的 pid 次序,错配就是一份行数对得上、内容全错的数据。
    tk_parts, K = [], None
    for f in files:
        cand = glob.glob(os.path.join(os.path.dirname(f),
                                      "topk*_" + os.path.basename(f)[len("verdict_"):]))
        if len(cand) != 1:
            return a, None
        k = int(os.path.basename(cand[0]).split("_")[0][len("topk"):])
        K = k if K is None else K
        if k != K:
            print(f"⚠️ top-k 的 K 不一致({K} vs {k}),跳过 top-k 分析")
            return a, None
        t = np.fromfile(cand[0], dtype=np.int32)
        n_main = len(np.fromfile(f, dtype=_REC))
        if t.size % k or t.size // k != n_main:
            print(f"⚠️ {os.path.basename(cand[0])} 行数 {t.size // k:,} 与主流 {n_main:,} 不符,跳过 top-k")
            return a, None
        tk_parts.append(t.reshape(-1, k))
    tk = np.concatenate(tk_parts) if tk_parts else None
    if tk is not None:
        print(f">>> top-k 侧流: K={K}, {len(tk):,} 行(与主流一一对应)")
    return a, tk


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
    ap.add_argument("--pairs", type=int, default=0,
                    help="按 (草稿给的 -> 目标要的) token 对分组,列出最高频的前 N 组")
    ap.add_argument("--pair-samples", type=int, default=3, help="每组打印几个带上文的例子")
    args = ap.parse_args()

    a, topk_arr = load(args.dir, args.tag)
    K = int(a["slot"].max()) + 1
    order = np.lexsort((a["slot"], a["step"], a["req"]))
    if topk_arr is not None:
        topk_arr = topk_arr[order]          # ★ steps() 会重排 a,top-k 必须同序重排,否则逐行对应就断了
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
    # 重建每个请求实际发出的 token 序列(⑥⑪ 共用)。规则:该步接受的草稿 token,然后要么是
    # 断点处的 target_tok,要么(全接受时)是 bonus_tok。★ 没有 bonus_tok 这一列,全接受的
    # 步会静默少一个 token,上文就整体错位 —— 这也是当初非加它不可的原因。
    stream: dict[int, dict[int, int]] = {}
    if (args.samples or args.pairs) and args.tokenizer and dec_raw is not None:
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

    def ctx_of(row: int) -> str:
        r, oi = int(a["req"][row]), int(a["out_idx"][row])
        d = stream.get(r, {})
        return dec_raw_many([d[k] for k in range(max(0, oi - args.ctx), oi) if k in d])

    if args.samples and args.tokenizer and dec_raw is not None:
        print("\n" + "=" * 78)
        print(f"⑥ 断点样例(每档 {args.samples} 个,上文 {args.ctx} 个 token)")
        print("=" * 78)
        rng = np.random.default_rng(args.seed)
        for lo_b, hi_b, name in BANDS:
            sel = br[(p1 > lo_b) & (p1 <= hi_b)]
            if not len(sel):
                continue
            pick = rng.choice(sel, size=min(args.samples, len(sel)), replace=False)
            print(f"\n{'─' * 78}\n【{name}】 共 {len(sel):,} 个断点,抽 {len(pick)} 个\n{'─' * 78}")
            for row in pick:
                r, oi = int(a["req"][row]), int(a["out_idx"][row])
                ctx = ctx_of(row)
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

    # ---------------------------------------------------------------- ⑩ oracle top-k 覆盖
    # 「目标要的那个词,排在草稿的第几位?」——【单路径 dump 能回答的最有价值的一个问题】。
    #
    # ★ 它能做什么:把「草稿知道但排错序」和「草稿根本不知道」分开。前者是蒸馏/损失问题
    #   (温度、double-norm 那条线),后者是容量/数据问题,下一步动作完全相反。
    #
    # ⚠ 它【不能】做什么:推不出 oracle 的 accept_len。oracle 换掉 slot s 的词之后,
    #   slot s+1..K-1 是草稿基于自己那个【错词】草出来的,前缀变了就得重草 —— 单路径 dump
    #   里没有那条分支。所以下面只给「每救回一个断点至少多接受 1 个 token」这个【下界】,
    #   不给 oracle accept_len。这也正是树形草稿实测收益总低于覆盖率数字的原因。
    #
    # ⚠ 而且 oracle 不是免费的:从 top-k 接受意味着目标要验证 k 个候选,正确的指标是
    #   「每个被验证 token 的接受长度」。否则 k=词表大小 就能"证明" accept_len=block_size。
    if topk_arr is not None:
        print("\n" + "=" * 78)
        print("⑩ 目标要的词排在草稿 top-k 的第几位(只看断点)")
        print("=" * 78)
        KT = topk_arr.shape[1]
        hit = topk_arr[br] == tt[:, None]                       # [n_break, K]
        found = hit.any(axis=1)
        rank = np.where(found, hit.argmax(axis=1), -1)    # 0-based;-1 = 不在 top-k 里

        n0 = int((rank == 0).sum())
        print(f"   自检:rank 0 的断点 {n0} 个 —— 必须是 0。"
              f"{'✅' if n0 == 0 else '❌ 不为 0 说明 top-k 与主流错位,下面的数不能信'}")
        print(f"   (断点处草稿的第 1 名【就是】它自己选的那个词,而那个词与目标不符才成为断点)\n")

        print(f"{'k':>6} {'覆盖率(累计)':>16} {'救回断点数':>12} {'Δaccept_len 下界':>18}")
        for k in (2, 4, 8, 16, 32, 64):
            if k > KT:
                break
            c = int(((rank >= 1) & (rank < k)).sum())
            print(f"{k:>6} {c/len(br)*100:>15.2f}% {c:>12,} {c/n_steps:>17.3f}")
        out_k = int((rank < 0).sum())
        print(f"{'>k':>6} {out_k/len(br)*100:>15.2f}% {out_k:>12,}"
              f"{'  ← 草稿【不知道】,容量/数据问题':>18}")

        print(f"\n{'排名区间':>12} {'断点数':>10} {'占断点':>9}   含义")
        for lo_r, hi_r, lab in ((1, 2, "第 2 名"), (2, 5, "第 3-5 名"), (5, 10, "第 6-10 名"),
                                (10, 32, "第 11-32 名"), (32, 64, "第 33-64 名")):
            if lo_r >= KT:
                break
            c = int(((rank >= lo_r) & (rank < min(hi_r, KT))).sum())
            note = "★ 知道,只是排错序 —— 蒸馏/损失能买到" if hi_r <= 5 else \
                   ("知道得很模糊" if hi_r <= 32 else "几乎等于不知道")
            print(f"{lab:>12} {c:>10,} {c/len(br)*100:>8.2f}%   {note}")
        print(f"{'不在 top-k':>12} {out_k:>10,} {out_k/len(br)*100:>8.2f}%   草稿不知道")

        # 按置信档拆 —— p>0.9 那一档才是高价值的,它的覆盖率决定训练方向
        print(f"\n{'置信档':>22} {'断点数':>9} {'覆盖@8':>9} {'覆盖@64':>9} {'不在 top-k':>11}")
        for lo_b, hi_b, name in BANDS:
            m = (p1 > lo_b) & (p1 <= hi_b)
            n = int(m.sum())
            if not n:
                continue
            c8 = int(((rank >= 1) & (rank < min(8, KT)) & m).sum())
            c64 = int(((rank >= 1) & m).sum())
            miss = (rank < 0)[m].mean() * 100
            print(f"{name:>22} {n:>9,} {c8/n*100:>8.2f}% {c64/n*100:>8.2f}% {miss:>10.2f}%")
        print("   ★ p>0.9 这一档的覆盖@64 是关键:高 = 草稿知道答案只是排不上去(蒸馏);"
              "低 = 真不知道(容量)。")

        # 按目标 token 类别拆 —— 数字那一类尤其想知道
        if args.tokenizer and dec_raw is not None:
            kcat10 = make_kind(dec_raw)
            ktt10 = np.array([kcat10(x) for x in tt])
            print(f"\n{'目标要的类别':>14} {'断点数':>9} {'覆盖@8':>9} {'覆盖@64':>9}   ")
            for c in sorted(set(ktt10.tolist())):
                m = ktt10 == c
                n = int(m.sum())
                c8 = int(((rank >= 1) & (rank < min(8, KT)) & m).sum())
                c64 = int(((rank >= 1) & m).sum())
                print(f"{c:>14} {n:>9,} {c8/n*100:>8.2f}% {c64/n*100:>8.2f}%")
            print("   ★ 数字那一行:覆盖高 = 草稿算得出、只是没押第一(可训);"
                  "覆盖低 = 草稿做不了算术(要么加容量,要么别指望它)。")

    # ---------------------------------------------------------------- ⑪ 按 token 对分组的例子
    # ④ 给的是「哪些 (A -> B) 最常断」的频次,但一行频次说不清【什么情况下】草稿爱选 A。
    # 这一段把同一个 (A -> B) 的若干次断点摆在一起,连上文一起看 —— 这才是人能读懂、能据此
    # 判断"这是不是一个真的失效模式"的形式。写简报、给别人看,用的是这一段。
    if args.pairs and args.tokenizer and dec_raw is not None:
        print("\n" + "=" * 78)
        print(f"⑪ 最高频的 {args.pairs} 组 (草稿给的 -> 目标要的),每组 {args.pair_samples} 个带上文的例子")
        print("=" * 78)
        pair_key = dt.astype(np.int64) * (1 << 21) + tt.astype(np.int64)
        uniq_p, cnt_p = np.unique(pair_key, return_counts=True)
        rngp = np.random.default_rng(args.seed)
        for i in np.argsort(-cnt_p)[: args.pairs]:
            d_id, t_id = int(uniq_p[i] >> 21), int(uniq_p[i] & ((1 << 21) - 1))
            rows_p = br[pair_key == uniq_p[i]]
            # 同组内的置信分布 —— 决定这组值不值得追
            pp = p1[pair_key == uniq_p[i]]
            hi_share = float((pp > 0.9).mean()) * 100
            print(f"\n{'─' * 78}")
            print(f"【{cnt_p[i]:,} 次 · 占断点 {cnt_p[i]/len(br)*100:.2f}% · 其中目标 p>0.9 的占 {hi_share:.0f}%】"
                  f"  草稿 {dec_raw(d_id)!r}  ->  目标 {dec_raw(t_id)!r}")
            print("─" * 78)
            for row in rngp.choice(rows_p, size=min(args.pair_samples, len(rows_p)), replace=False):
                print(f"  上文 …{ctx_of(int(row))!r}")
                print(f"       目标 top1 p={float(a['target_top1_p'][int(row)]):.3f}"
                      f"   目标给草稿那词 p={float(a['target_p_draft'][int(row)]):.3f}")
        print("\n   ★ 同一组里几个例子的上文如果长得像 -> 是可命名的失效模式;各不相干 -> 只是高频词碰撞。")

    # ---------------------------------------------------------------- ⑫ 语义严重度
    # ⑤/⑦ 的类别是【字符形态】的,「英文词/词片」里混着 the/is/are 这种虚词。
    # 这一段按【语义承载】重切:草稿猜错一个 the,和把 32 算成 24,是完全不同的病。
    # ⚠ 但两者对吞吐的损失【完全相同】—— 断点就是断点。本段只回答"模型哪里真的不懂",
    #   不用于决定优化优先级(那要看 ⑦ 的体量和 ⑩ 的排名)。
    if args.tokenizer and dec_raw is not None:
        print("\n" + "=" * 78)
        print("⑫ 按【语义严重度】重切断点 —— 哪些是「意思都不对了」")
        print("=" * 78)
        kc12 = make_kind(dec_raw)

        def word(tid):
            return dec_raw(int(tid)).strip().lower().strip(".,:;!?'\"()[]{}")

        def numval(tid):
            w = dec_raw(int(tid)).strip().replace(",", "")
            try:
                return float(w)
            except ValueError:
                return None

        def tier(di, ti):
            kd_, kt_ = kc12(di), kc12(ti)
            if kt_ == "特殊标记" or kd_ == "特殊标记":
                return "T1 控制 token(终止/回合边界)"
            if kt_ in FORMAT_KINDS and kd_ in FORMAT_KINDS:
                return "T0 纯格式(两边都是标点/空白)"
            nd, nt = numval(di), numval(ti)
            if nd is not None and nt is not None:
                return "T3a ★ 数值不同(算错/抄错)" if nd != nt else "T0 纯格式(两边都是标点/空白)"
            wd, wt = word(di), word(ti)
            d_fn, t_fn = wd in FUNC_WORDS, wt in FUNC_WORDS
            d_ct = bool(wd) and not d_fn and kd_ in ("英文词/词片", "混合", "数字")
            t_ct = bool(wt) and not t_fn and kt_ in ("英文词/词片", "混合", "数字")
            if t_ct and d_ct:
                return "T3b ★ 实词换实词(说成了别的意思)"
            if t_ct or d_ct:
                return "T3c ★ 实词 ↔ 虚词/格式(该说实义时没说,或反之)"
            if d_fn and t_fn:
                return "T2 虚词/语法(the/is/are 之类)"
            return "T2 虚词/语法(the/is/are 之类)"

        tiers = np.array([tier(d_, t_) for d_, t_ in zip(dt, tt)])
        order12 = ["T3a ★ 数值不同(算错/抄错)", "T3b ★ 实词换实词(说成了别的意思)",
                   "T3c ★ 实词 ↔ 虚词/格式(该说实义时没说,或反之)",
                   "T2 虚词/语法(the/is/are 之类)", "T1 控制 token(终止/回合边界)",
                   "T0 纯格式(两边都是标点/空白)"]
        print(f"{'严重度':>34} {'断点数':>9} {'占断点':>8} {'其中 p>0.9':>11} {'该档高置信率':>13} {'Δ 下界':>8}")
        sev_rows = 0
        for name in order12:
            m = tiers == name
            n = int(m.sum())
            if not n:
                continue
            hi = int(((p1 > 0.9) & m).sum())
            if name.startswith(("T3",)):
                sev_rows += hi
            print(f"{name:>34} {n:>9,} {n/len(br)*100:>7.2f}% {hi:>11,} {hi/n*100:>12.1f}% {hi/n_steps:>+8.3f}")
        other = int((~np.isin(tiers, order12)).sum())
        if other:
            print(f"{'(未归类)':>34} {other:>9,}")
        print(f"\n   ★ T3 三档合计的高置信断点 = {sev_rows:,},Δaccept_len 下界 {sev_rows/n_steps:+.3f}")
        print("   —— 这才是「主模型笃定、而草稿说成了别的意思」的那部分。")
        print("   ⚠ T0/T2 对吞吐的损失与 T3 完全相同(断点就是断点),只是病因不同,别据此砍优化项。")

        # 每个 T3 档给例子
        if args.pair_samples:
            rng12 = np.random.default_rng(args.seed)
            for name in order12[:3]:
                idx = np.flatnonzero((tiers == name) & (p1 > 0.9))
                if not len(idx):
                    continue
                print(f"\n{'─' * 78}\n【{name}】 高置信档 {len(idx):,} 个,抽 {min(args.pair_samples, len(idx))} 个\n{'─' * 78}")
                for j in rng12.choice(idx, size=min(args.pair_samples, len(idx)), replace=False):
                    row = int(br[j])
                    print(f"  上文 …{ctx_of(row)!r}")
                    print(f"       草稿 -> {dec_raw(int(dt[j]))!r}      主模型 -> {dec_raw(int(tt[j]))!r}"
                          f"   (主模型 top1 p={float(p1[j]):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
