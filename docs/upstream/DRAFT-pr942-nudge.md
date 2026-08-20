# DRAFT — nudge on PR #942 (NOT SENT; for review)

> 状态:草稿。#942 目前 mergeable ✓、请求的验证已答复、冲突已解,最后更新 2026-08-16,静默 5 天。
> 请求审阅人:orestis-z / rahul-tuli / fynnsu / shanjiaz。评审意见:尚无。

---

@shanjiaz — since you're looking at #952 anyway, would you mind casting an eye over #942? It
is small (227 lines, no deletions) and I think everything asked for has been answered:

- @eldarkurtic asked for runtime and accuracy validation. Measured on DeepSeek-V4-Flash across
  8 Ascend NPUs: **no measurable runtime cost** (~20–50 ms on ~2080 ms steps) and acceptance
  length **unchanged** across five datasets (mean difference −0.0002).
- The merge conflict from the #871 overlap is resolved; it is mergeable again.

It is a correctness fix rather than a feature: with uneven token counts across ranks the
per-rank mean-of-ratios is not the token-weighted objective it is meant to be, and the
all-reduced denominator restores it. On balanced runs it is a no-op, which is also what the
tests pin down.
