# DSV4 DSpark 草稿模型 — 结构与前向速记

> 归档自架构讨论。行号对应 `src/speculators/models/dsv4_dspark/` 与 `src/speculators/models/dflash/`
> 当时的快照(近期若改过 `load_verifier_layer` 等,前向部分行号可能整体平移十余行,结构不变)。

记号:`H`=hidden_size,`hc`=hc_mult(超连接流数,默认 4),`E`=256 routed experts,
`A`=anchor 数(MAX_ANCHORS),`B`=block_size,`A×B`=draft 预测槽位数(代码注释里叫 `TB`),
`T/L`=上下文序列长,`γ`=一块里的投机位数(=block_size−1)。

---

## 1. 草稿模型结构(standalone)

```
 DSV4 DSpark 草稿模型 (draft)        H=hidden_size    hc=4 条超连接流    E=256 routed experts

 ① target hidden [·,3H] ─fc(3H→H)─► hidden_norm ─► main_x 上下文 ─┐  ← 被 attend,不流过 block
                                                                 │
 ② token(anchor+MASK) [·,L] ─embed─► streams [·,hc,H] ─┐         │  ← MASK 占位,流过 block
                                                       ▼         ▼
                        ┌──────────────── DecoderBlock × 3 ───────────────┐
                        │ attn_hc → attn_norm                             │
                        │ MLA(+sink):  q = streams,注意 [历史:滑窗因果 | 本块:非因果]
                        │ ffn_hc → ffn_norm → MoE(router top-k/E + shared) │
                        └────────────────────────┬────────────────────────┘
                                    hc_head(hc→1) → norm → hidden
                                                 │
                           ┌──────────────────────┼──────────────────────┐
                        lm_head               Markov head            Confidence head
                        base logits           → markov_bias          → accept prob [·,L]
                           └───────(+ markov_bias)───────► draft logits [·,vocab]      (旁支 / detach)
```

**两个输入,角色不同:**

- **① target hidden**(verifier 3 层拼接 `[·,3H]`):经 `self.fc`(main_proj **3H→H**)+ `self.hidden_norm`
  → `main_x` 上下文。它当 **MLA 里被 attend 的上下文(KV/记忆),本身不流过那 3 层 block**。
- **② token ids**(每块 = `[anchor 真 token, MASK, MASK, …]`,MASK id=128799):经 `embed_tokens` →
  `noise_embedding` → 展开成 `hc` 条超连接流 `streams`。**这才是流过 3 层 block、被预测的主体。**
  （大部分是 MASK 占位,只有每块 slot0 是真 anchor token → 不泄漏答案。）
- 两者在 **MLA** 里相遇:`streams` 当 query,去 attend `[main_x 上下文 | 本块]`。

**每层 DecoderBlock**(`backbone/block.py`):mHC 包住两件事 —
`attn_hc → attn_norm → MLA(+sink) → place`,再 `ffn_hc → ffn_norm → MoE → place`;超连接处做 Sinkhorn 投影。

**收尾**:`hc_head`(HyperHead,`hc` 条流并回 1 条)→ `norm` → `hidden` →
`lm_head`(base logits)→ Markov head 加 `markov_bias` 得 **draft logits**;Confidence head 旁支(输入 detach)预测每位置接受概率。

---

## 2. 注意力连接:块内非因果 / 块间因果

对着 `dflash/attention.py` 的 `or_masks(base_prefix_mod, same_block_mod)`:

```
        历史上下文(滑窗 W 内的过去 token)
                    ▲
                    │ 回看(只能往上/过去)
        ┌───────────┴──────── block_n ────────┐
        │   a ⇄ s1 ⇄ s2 ⇄ s3 ⇄ s4             │   块内双向 = 非因果 → γ 个 MASK 一次并行填
        └───────────┬─────────────────────────┘
                    │  block_{n+1} 能回看 [历史 + block_n];
                    │  block_n 看不到 block_{n+1}(未来)   ⇒ 块间因果
        ┌───────────▼──────── block_{n+1} ────┐
        │   a ⇄ s1 ⇄ s2 ⇄ s3 ⇄ s4             │   块内双向 = 非因果
        └─────────────────────────────────────┘
```

- **本块内**(`same_block_mod`,`non_causal=True`):γ 个 slot 互相双向 attend = **块内非因果**
  → 一次前向**并行填出整块**(diffusion 味,但 single-step,不是多轮扩散)。
- **对历史/上下文**(`base_prefix_mod`):只看 anchor **左侧** + 滑窗 W、同文档 = **因果 + 滑窗**;
  看不到未来 ⇒ **块与块之间也因果**。
- **MLA ⟂ mask**:MLA 只管**怎么算**(低秩 q `wq_a→q_norm→wq_b` + 单个共享 KV `wkv→kv_norm` +
  per-head `sink` + 低秩输出 `wo`);上面两条只是喂进去的 `attn_bias` mask。
- **per-head sink**:softmax 分母里多加一个永远在场的合成 key(`exp(sink_h)`),不贡献 value —
  当窗口里有效 key 很少(如第一块)时稳住注意力。

---

## 3. 训练前向:两条 hidden 流 + anchor 如何放大 A×B 轴

训练时 `_backbone_forward`(`dsv4_dspark/core.py`)吃**两份** verifier 抽头:

```
  ①  hidden_states  [1, T, 3H]                     ②  verifier_last_hidden_states  [1, T, H]
  = 中间 [40,41,42] 三层拼接 (draft 的输入/上下文)     = 最后一层过完 final norm (拿来当 label)
        │                                                │
   self.fc + hidden_norm       (531-532)          verifier_lm_head            (560)
        │                                                │
   fc_output [1, T, H]  上下文,长 T                verifier_logits [1, T, vocab]  长 T
        │                                                │
        │   anchor_positions [A] ── 从 T 里挑 A 个位置 ──►│  (select_anchors)
   + noise_embedding [1, A×B, H]  ★  (529)               │
        │                                    targets = verifier_logits[anchored]  [1, A×B, vocab] ★ (566)
   ★ for layer in self.layers (×3)  (575)               │
        │                                                │
   hc_head → norm → lm_head    (631-632)                │
        │                                                │
   logits [1, A×B, vocab]  ★  ──────► compound_loss ◄────┘   (metrics.py 460)
                                       对 A×B 个 slot 求 CE+TV;分母=有效 slot 数,不含 padding (528)
```

- **① 是 student 输入**(走完整 draft,回传);**② 只过 verifier head、`no_grad`,造 label**(不回传)。
- **`★` = 随 anchor 数 A 变大的 `A×B` 轴**(`noise/streams/draft/logits/targets/loss` 槽位);
  `T`(上下文)、`fc_output`、`verifier_logits` 整条 T **不变**。
- anchor 设太多 ≠ 更多监督(`select_anchors` 随机子采样,超序列长的部分是 masked padding,不进 loss/分母);
  真实代价是 **`A×B × vocab` 的 tv-softmax 显存线性涨 → OOM/逼小 batch → 间接掉质量**。按不 OOM 的最大值设即可。
- `double-norm`(`DSPARK_TEACHER_DOUBLE_NORM=1`)只污染 **label 那一支**(526-528),碰不到 draft forward。

---

## 4. 关键约定

- **aux 抽头(①)= POST-layer、含完整残差**;DSV4 的 `hc_post` 已折入残差,所以裸 `hidden_states.mean(1)` 是对的。
- **`sample_from_anchor=True`**(DSpark 默认,匹配 vllm-ascend serve):slot k 预测 pos k+1,slot0 也训练。
- **noaux_tc 负载均衡**:`for layer` 之后每步更新一次 router 选择 bias(`DSPARK_MOE_BALANCE=1`);
  bias 只改 selection、改不到 combine 权重与 router 权重 → 只能均衡 count、修不动质量塌缩(见负载均衡记录)。
