#!/usr/bin/env python3
"""这份训练语料的 response,真的是目标模型贪心吐出来的吗?

WHY
---
2026-09-20:`hs_self_oracle.py` 已经证明 HS dump 的机制是对的(serve 自己吐的 token,
dump 的 hidden 过 lm_head 能 100% 重现;回灌自产文本 mismatch 0.00%)。但同一套 HS 在
`arrow_0730_77w_dedup` 的 **response 段**(loss_mask==1)上 mismatch 高达 **64%**。

两件事都成立,只剩一个解释:**这份语料的 response 不是目标模型贪心生成的。**

这件事的份量远超那个算子故障 —— `experiments/STATUS.md` 显示所有 DSpark 训练都跑在这份
Arrow 上。如果 response 不是目标自己的输出,草稿一直在蒸馏一个**目标不会产生的分布**,
而 DSpark 的全部价值就建立在「草稿预测目标会说什么」上。已发布草稿 4.42 而我们停在 4.18,
这可能是其中一块。

所以别再绕着 HS 转,直接问语料:**把 prompt 喂回去贪心续写,和语料里的 response 比。**
不涉及 HS、不涉及 lm_head、不依赖任何中间约定 —— 只有「模型会不会这么说」。

同时验一件我一直在假设的事:`loss_mask==1` 到底标的是不是 response。脚本会打印它的
分布(首个 1 的位置、占比),假设错了一眼就能看见。

USAGE
-----
    ENDPOINT=http://localhost:7000/v1 \
    ARROW=/home/canada_group_folder/dataset/arrow_0730_77w_dedup \
      python corpus_provenance_check.py --n 8 --gen 32

读法
----
    前缀完全一致长度 ≈ --gen           ⟹ 语料就是本模型贪心输出,语料没问题
    第 1~2 个 token 就分叉,逐 token 一致率低  ⟹ 语料【不是】本模型贪心输出
      → 可能是:别的模型/别的 pin 生成的、采样不是贪心、或者清洗/去重步骤改动了内容
      → 这会同时解释 HS 的 64%,而且意味着训练的教师信号一直是错的
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "http://localhost:7000/v1"))
    ap.add_argument("--arrow", default=os.environ.get("ARROW"))
    ap.add_argument("--n", type=int, default=8, help="查几行")
    ap.add_argument("--gen", type=int, default=32, help="每行贪心续写几个 token")
    ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--max-prompt", type=int, default=2048, help="prompt 超过就截(从尾部保留)")
    ap.add_argument("--id-base", type=int, default=960000,
                    help="★ 必须 > 数据集行数,否则会污染生产 HS 目录")
    ap.add_argument("--rows-full", type=int, default=772684)
    args = ap.parse_args()

    if not args.arrow:
        raise SystemExit("!! 需要 --arrow(或 ARROW=)")
    if args.id_base <= args.rows_full:
        raise SystemExit(f"!! --id-base {args.id_base} 落在数据集行号内,会把生产 HS 目录写脏。")

    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

    import openai  # noqa: PLC0415
    from datasets import load_from_disk  # noqa: PLC0415

    ds = load_from_disk(args.arrow)
    if hasattr(ds, "keys") and not hasattr(ds, "num_rows"):
        ds = ds[next(iter(ds.keys()))]
    try:
        ds = ds.with_format(None)
    except Exception:  # noqa: BLE001
        pass

    cli = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
    model = cli.models.list().data[0].id
    print(f"serve={model}   arrow={args.arrow}  {len(ds)} 行\n")

    has_mask = "loss_mask" in ds.column_names
    if not has_mask:
        print("⚠ 这份 Arrow 没有 loss_mask 列,只能按「后半段」切 prompt/response。")

    tot_match = tot_cmp = 0
    prefix_lens = []
    for i in range(args.n):
        row = ds[args.start_row + i]
        ids = list(row["input_ids"])
        if has_mask:
            m = list(row["loss_mask"])
            ones = [j for j, v in enumerate(m) if v]
            if not ones:
                print(f"  [{i}] loss_mask 全 0,跳过"); continue
            k = ones[0]
            frac = len(ones) / len(m)
        else:
            k = len(ids) // 2
            frac = 0.5
        prompt, resp = ids[:k], ids[k:]
        if len(prompt) > args.max_prompt:
            prompt = prompt[-args.max_prompt:]
        want = min(args.gen, len(resp))
        if want < 2 or not prompt:
            print(f"  [{i}] 太短,跳过"); continue

        r = cli.completions.create(
            model=model, prompt=prompt, max_tokens=want, temperature=0,
            extra_headers={"X-Request-Id": f"hs_{args.id_base + i}"},
            extra_body={"return_token_ids": True}, timeout=600,
        )
        got = getattr(r.choices[0], "token_ids", None)
        if got is None:
            d = r.choices[0].model_dump()
            got = d.get("token_ids")
        if not got:
            print(f"  [{i}] serve 没返回 token_ids,跳过"); continue
        got = [int(t) for t in got][:want]
        tgt = [int(t) for t in resp[:want]]

        pref = 0
        while pref < len(got) and pref < len(tgt) and got[pref] == tgt[pref]:
            pref += 1
        same = sum(1 for a, b in zip(got, tgt) if a == b)
        tot_match += same; tot_cmp += len(tgt)
        prefix_lens.append(pref)
        print(f"  [{i}] len={len(ids):>5} prompt={k:>5} (loss_mask 1 占比 {frac:.2f})  "
              f"前缀一致 {pref:>3}/{want}   逐 token 一致 {same}/{len(tgt)} "
              f"({100.0 * same / len(tgt):.1f}%)")
        if pref < 4:
            print(f"        语料: {tgt[:8]}")
            print(f"        模型: {got[:8]}   ← 第 {pref} 个 token 就分叉")

    print("\n" + "=" * 70)
    if not tot_cmp:
        print("⚠️ 一条都没比成,看上面的逐条输出。"); return 1
    rate = 100.0 * tot_match / tot_cmp
    mean_pref = sum(prefix_lens) / max(len(prefix_lens), 1)
    print(f"逐 token 一致率 {rate:.1f}%   平均完全一致前缀 {mean_pref:.1f} 个 token "
          f"(上限 {args.gen})")
    print("=" * 70)
    if rate > 90:
        print("✅ 语料就是本模型的贪心输出 —— 语料没问题。")
        print("   那 HS 在 response 段 64% 就还没解释,回去查 HS(但 self-oracle 已证明机制对,")
        print("   所以更可能是我算 response 段的方式有问题)。")
    elif rate > 50:
        print("⚠️ 部分一致 —— 像是同一个模型但不同栈/不同精度/不同批次组成造成的漂移。")
        print("   影响有限但不可忽略;确认 rollout 当时的 pin 和采样参数。")
    else:
        print("❌ **语料不是本模型的贪心输出。** 这会同时解释 HS 在 response 段的 64%。")
        print("   份量:experiments/STATUS.md 显示所有 DSpark 训练都跑在这份 Arrow 上 ——")
        print("   草稿一直在学一个目标模型【不会产生】的分布,而 DSpark 的全部价值就建立在")
        print("   「草稿预测目标会说什么」上。这可能是已发布草稿 4.42 而我们停在 4.18 的一块。")
        print("   下一步:查 docs/deployment/ascend-npu-dsv4-rollout-data.md 确认这份 77W 的")
        print("   生成栈与采样参数,以及 clean/dedup 步骤有没有改动过 response 内容。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
