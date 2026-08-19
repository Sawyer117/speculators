# DSV4-DSpark 提升试验工作日志

**这个文件不在主线上,也不该合进主线。** 主线 worklog
(`ascend-npu-dsv4-worklog.md`)记录的是把 5-epoch 交付权重做出来的那条线;这里记录的是**在
那份权重之上继续找收益**的试验——它们多数会失败,失败本身才是要留下的东西。

约定与主线一致:**只追加,不改写**。每条记录必须能回答三个问题:配置是什么、结论是什么、
以及**这个结论能不能被测量**——最后一条是本文件反复踩到的坑。

---

## 2026-08-19 — Correction 头(TYS5537/dspark_next 特性集):训练侧优势单调收窄,主动中止

### 配置

`ckpt_faithful_ep_20260818_122129` / 日志 `faithful_ep_20260818_122129.log`,启动命令:

```bash
DSPARK_EP=1 BF16_EXPERTS=1 RECOMPUTE=1 COMPILE=0 \
DSPARK_MOE_BALANCE=1 DSPARK_MOE_BALANCE_RATE=1e-3 DSPARK_LOG_EXPERT_LOAD=1 \
INIT_LAYER=1 INIT_MOE_NO_ROUTER=1 \
LR=2e-4 EPOCHS=5 MAX_ANCHORS=512 CKPT_FREQ=0.5 \
DATA=/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow_0730_77w_dedup \
  bash examples/ascend_npu_dflash/train_dsv4_dspark_correction.sh faithful
```

与主线 BEST 命令(worklog 2026-08-04)**逐字段相同**,只多一个 `DSPARK_LOG_EXPERT_LOAD=1`
(纯日志)并换了 wrapper。⚠ **没有** `DSPARK_GLOBAL_LOSS_REDUCE=1` ⟹ 本次是 #942 的 **OFF 臂**,
因此对照基线取同为 OFF 臂的 `ckpt_faithful_ep_20260804_165215`(`ep5p0-ropefix`),而不是
lossreduce 那条——两臂的 loss 归一化方式不同,逐步 diff 会把两个变化搅在一起。

特性侧取自 collaborator 的 "his best config",维度/LR/损失/词表/几何全部用我们的:
`--correction-hidden-size 1024`(他是 512 @ hidden 2560;我们 hidden 4096,按比例放大,
8 头正好 128/头)、`--correction-rank 256`(= released draft 在我们这个规模上的 `markov_rank`)、
`--correction-output-mode logits`、hidden-aux-loss 0.1、hidden-feedback、project-corrected-hidden、
with-markov;三个此前是死代码的 `--dflash-*` backbone 旗标经 `_backbone_forward` 补丁后生效。
⚠ 一处刻意的偏离:`--no-confidence-detach-features`(本 fork 默认 detach,ledger 里每个检查点
都是 detach 训的)。

### 结论:优势单调收窄进噪声带

同种子、同数据 ⟹ batch 序列逐步一致,这是**配对比较**,灵敏度远高于 eval 侧 ±0.025 的噪声带。

| 步数 | accept_len Δ (CORR − ROPEFIX) | loss Δ | 分析器判定 |
|---:|---:|---:|---|
| 17,077 | **+0.038** | −0.014 | ✅ better |
| 24,262 | +0.027 | −0.009 | ✅ better |
| 25,879 | +0.023 | −0.007 | ✅ better |
| **38,903** | **+0.017** | −0.007 | **→ ~same**(< 0.5% 阈值) |

四点单调,两个指标同向。**诚实的读法不是"Correction 头没用",而是"它给了更快的起步,
但两条曲线正在收敛到同一渐近线"**——至少在训练侧 soft accept_len 这个口径上。

**中止于 step 38,903(1.56 epoch,34.4 h)。** 保留的检查点:**1.0ep 在 `/0`、1.5ep 在 `/1`**
(0.5ep 已被 1.0ep 覆盖,符合 CKPT_FREQ=0.5 的目录轮转)。丢掉的只有 1.5ep 之后的 1,557 步。

### ★ 为什么中止:这个实验我方无法评测

这是本条记录最该被后来人看到的部分。查证结果:

```
src/speculators/models/dsv4_dspark/weights.py         "correction" 出现 0 次(只认 markov_head / confidence_head)
vllm-ascend vllm_ascend/models/deepseek_v4_dspark.py  "correction" 出现 0 次
```

⟹ 转换出的权重会**丢掉整个 Correction 头**,而该模型的推理路径是
`LMHead(h + Δh) + Δlogits + markov_bias`——少了 Δh 和 Δlogits 就是训练/服务不匹配,
测出来的数没有意义。**训练侧指标是这次 run 能给我们的全部信号。**

在"优势已收窄到判不出"+"我方测不了"这两条同时成立时,再烧 3.5 天 × 24 卡去换一个
测不到的结果不划算,故中止。1.0ep / 1.5ep 两份权重交给有 Correction 头推理实现的一方评测。

### 附带确认(与本试验无关但同批观测)

- **#942 的两层拆分得到再次验证。** 各卡累计监督 token 现在全在 ±0.1% 内,前后半程偏差
  相关性 **−0.535**(前一次 +0.588)⟹ 方向随机、互相抵消,累计不均衡在这棵树上确实随
  #867 消失了。但 **366/38904 步仍出现整卡零监督 token**,逐步偏差照旧存在。
  ⟹ #942 的论据重心正式从"修一个可测的持续偏差"移到**目标函数定义本身的正确性**。
- **MoE 有效专家数偏高但健康**:99 / 104 / 92,对比 ropefix 收官的 62 / 73 / 81;但三层并集
  全饱和(251/255/256),逐位置准确率全线上行(p5 0.4885 → 0.5455),**无尾部坍塌迹象**。
  记录为观察项,非问题。
- 日志里 `NOTABLE` 那句 "recompile spikes eat 34% of wall-clock" 是**旧口径误标**;
  以 `BOTTLENECK BREAKDOWN` 为准:真 recompile 1.3%,34% 是服务侧(straggler 28.8% + fetch 5.6%)。

---

## 2026-08-19 — DFlash2 的选择头寸:两个探针就位,尚未测量

### 来源与该信什么

inco.ai 的 DFlash2 博文在 DFlash 上加了两个模块。对我们**唯一有价值**的是它那张 recall 表
(Qwen3-4B / GSM8K):position 6 的 **recall@1 = 72.9% 而 recall@16 = 87.8%**,accept_len
**4.27(argmax)→ 6.79(top-16 oracle)**。也就是说尾部丢的 token 多半**还在候选里**,是
`argmax` 把它扔了。

⚠ **它的绝对数字不可迁移**:不同模型、**温度 1.0**(我们全线贪心;它自己的表里 DFlash 从
T=0 的 4.27 掉到 T=1 的 3.78),而且它的 DSpark 基线在自家 27B 表上**低于 Qwen 原生 MTP**
(3.62 vs 4.28),可信度存疑。**"DFlash2 打赢 DSpark"这个结论不引用。**

一处反推值得记:它 Table 2 里那个 "+DSpark correction, +77.8M, +9.6% latency" ——
`2 × 151936(Qwen3 词表)× 256 = 77.79M`,严丝合缝 ⟹ **那是 DSpark 的 Markov 头**,
不是 dspark_next 的 Correction 头。所以该表说的是"2M 的路径选择器打赢了 77.8M 的 Markov 头",
而 Markov 头是我们**已经背着**的(我们的是 129280×256×2 = 66.2M)。

### 与我们 num_spec=7 结论的关系:正交,不是替代

ns7 实验把丢分定位在训练未覆盖的 pos5/pos6,结论是"重训 block 8"。DFlash2 指出的是**同一
症状的第二个成因**:那些位置上正确 token 可能仍在 top-k 内,失败的是选择而非建模。
**两条是正交的**,不互相取代。

### Path Selection 的机制(便于日后不必重读原文)

候选格 K × 16,打分**可分解为相邻位置的两两项**,故 Viterbi 可解,O(K·16·16):

```
S_t(a, b) = U_t(b) + ⟨ A(a) ⊙ H(h_t), B(b) ⟩
```

`U_t(b)` = 草稿自己的 logit(一元项);后一项是 a→b 的低秩双线性兼容度(256 维,A 为前驱
身份、B 为后继身份),`H(h_t)` 是由隐藏状态算出的**逐维门控**,使"哪两个 token 搭"随上下文
变化。只留一条路径,不是树投机。

★ **对我们最关键的观察:这条一阶链我们已经有了。** 服务端
`_sample_sequential` 每步算的正是
`base_logits[:,t] + markov_bias(markov_embed(prev_ids))` —— 一元项 + 转移项俱全,而且同样是
rank-256 低秩分解。差别只在:**我们每步 argmax 立刻提交,它先算完所有 16×16 转移再联合解码**。
⟹ 存在一个**零新增参数、零训练**的中间实验:把贪心循环换成对现有 Markov 链的 Viterbi。
成本也低——不需要 materialize `[num_reqs,16,V]`,因为
`bias(a)[b] = markov_w2.weight[b] · markov_w1(a)`,取 16 个候选列就是 `[16,256]×[256,16]`。
(⚠ 该简化依赖 `markov_head_type=vanilla`,我们的配置正是。)

未公开的部分:selector 的训练方法、DP 递推、2.0M 参数的分配、T>0 拒绝采样的提议分布。
⟹ 真做是**重新推导**,不是复现。分支名 `dflash2-reproduce` 系历史命名,名不副实。

### 入图可行性:问题问错了,那段代码本来就在图外

在 `vllm-ascend` `pr-12006 @ 386530d1`(= eval ledger 记录的服务 build)上逐行确认:

```
:155  self._runnable = ACLGraphWrapper(self._run_model_from_graph_buffers, ...)   # 只包草稿 forward
:775  hidden_states = self._runnable()        ← 图重放到此结束
:779  return self._sample_sequential(...)     ← 已在图外
:898  dummy_run(捕获路径)只调 self._runnable(),从不调 _sample_sequential
```

旁证:`_propose` 体内有 `int(num_reqs_across_dp.max().item())` 这个 D2H 同步,捕获期非法。
⟹ **path selection 落在既有的 eager 段,不需要入图,也不会破坏任何已捕获的图。**

与"自适应草稿长度掉出图"的对照成立:K 直接进图的 key 与 buffer 形状
(`num_input_tokens = model_num_reqs * self.block_size`),K 一变全部重录;path selection
不碰 K / num_reqs / 任何 buffer 形状。**二者性质不同。**

⚠ **真正的成本转移到 host 侧 kernel launch**:`_sample_sequential` 是 eager 的,path selection
要在现有 K 步循环上再叠 K×(129280 全词表 topk + gather + bmm + max/argmax)。而 conc1 已证明
**投机一步 = AR 一步的 1.85×,卡住加速比的正是步开销**。⟹ 即使头寸为真,仍须过第二关:
涨的 accept_len 能否盖过加的步开销。

可复用情报:`topk` **已在捕获图内运行**(MoE 路由 `experts_selector.py`);已知限制仅
"torchair GE 图下 bf16 topk 不支持",解法 `.to(float32)`。

### 已就位、尚未运行的两个探针

| | 分支 / 文件 | 测什么 | 相对真实 |
|---|---|---|---|
| 训练侧 | speculators `dflash2-reproduce` `examples/ascend_npu_dflash/recall_headroom_probe.py` | 逐位置 recall@k + `oracle_accept_len` | **上界**(teacher forcing) |
| 服务侧 | vllm-ascend `dflash2-reproduce` `DSPARK_TOPK_DUMP` + speculators `topk_headroom_join.py` | 首错位置的 recall@k + 名次分布 | **下界**(只见得到第一次失配) |

**判据在看到数字之前就钉死**:pos≥4 的 recall@16 比 recall@1 高 **≥10 点**(或首错 recall ≥50%)
⟹ 头寸为真,值得算成本;**<3 点**(或 ≤20%)⟹ 草稿是真不知道那个 token,而非选错,
**path selection 对我们无效,当场结案**,只剩 block-8 重训。

该不对称性使筛查顺序明确:**训练侧探针的负结果是决定性的**(最有利条件下都没头寸),
正结果只是"值得再花服务端那一次确认"。

⚠ 训练侧探针是 teacher-forced 的:`dspark/core.py:159` `prev_token_ids = block_tokens` 喂给
Markov 头的是**真**前驱 token,而服务端喂的是草稿自己的选择。

两份脚本的数学都做过合成数据自测(recall@k 恰等于"放置名次 < k";recall@1 与 argmax 准确率
逐位相等;oracle_accept_len = 1 + top-k 内最长前缀;join 脚本对植入的 75.7%/rank-3 真值精确反解)。

### 运行前欠的一件事

`examples/ascend_npu_dflash/train_dsv4_dspark.sh:274` 把 `scripts/train.py` 写死了,训练侧探针
需要一个 `TRAIN_PY` 覆盖(一行,默认值不变)。跑法见脚本 docstring:**`LR=0` 是安全阀**——
AdamW 的 decoupled weight decay 同样按 lr 缩放,lr=0 时没有任何参数会动,续训权重不会被损坏。
