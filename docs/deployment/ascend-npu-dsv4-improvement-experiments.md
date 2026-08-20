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

### 交付给第三方(复核 / 自行评测所需的全部信息)

本条记录与实现代码**分处两个分支**,只给文件名对方跑不起来。完整清单:

| 要素 | 位置 |
|---|---|
| **实现代码** | `Sawyer117/speculators` 分支 **`feat/dspark-next-port`** @ `1d5827d`。自 `feat/dsv4-dspark` 分出,含 `78a0776` 合入 TYS5537/speculators@dspark_next,以及其后 6 条我们的适配提交(见下) |
| **启动脚本 + PROVENANCE** | 同分支 `examples/ascend_npu_dflash/train_dsv4_dspark_correction.sh`,文件头 10–52 行逐条写明**拿了什么 / 没拿什么 / 每条为什么** |
| **训练权重** | `run/ckpt_faithful_ep_20260818_122129/0`(1.0 epoch)与 `/1`(1.5 epoch)。**EP 分片的 DCP 格式**,非 safetensors;要用 speculators 的转换器转成部署格式,但见下方 ⚠ |
| **本次日志** | `run/faithful_ep_20260818_122129.log` |
| **对照基线日志** | `run/faithful_ep_20260804_165215.log`(`ep5p0-ropefix`,#942 的 OFF 臂,与本次同种子同数据) |
| **对照命令** | `python3 examples/ascend_npu_dflash/analyze_train_run.py <本次log> --baseline <基线log> --label CORRECTION --baseline-label ROPEFIX --out <目录>` |
| **环境** | `examples/ascend_npu_dflash/install_npu_env_dspark.sh` 是唯一权威安装脚本(torch/torch-npu 2.10.0、vLLM 0.23.0、vllm-ascend 从源码编) |

我们在他的分支之上做的 6 条适配(都是**让它在 DSV4 上跑起来**所必需,不改他的算法):

```
ef8cf7d  detach 覆盖挪到真正的 config 类;EXTRA_ARGS 透传
16db370  让三个 --dflash-* backbone 旗标在本模型上真正生效(此前是死代码)
92311d2  Correction 头宽按我们的 hidden 缩放(512@2560 -> 1024@4096),rank 保持发布值 256
f48572b  安装并解析 hs_connectors(上游同步后成为强依赖)
d5a9208  让 dsv4_dspark 通过 DSpark 的特性门禁
bcdc2d7  1-D 门控参数 —— FSDP2 拒绝标量
7c25438  昇腾上不走融合 Triton 损失
```

⚠ **评测必须由对方做,原因见上一节**:我们的转换器与 vllm-ascend 的模型侧对 `correction`
均零认知,转出来会丢掉整个 Correction 头。对方若有该头的推理实现,则:
① 用他自己的转换路径处理 `markov_head` / `confidence_head` **以及 correction 相关权重**;
② 我们这两份权重只训到 1.0 / 1.5 epoch(基线是 5.0 epoch 的),**跨 epoch 不可直接比**——
要比就与 `ep5p0-ropefix` 的 **1.0ep / 1.5ep 检查点**比,那两个的全量集五项平均分别是
**4.0788 / 4.1800**(见 eval ledger 的 FULL-SET 批次)。

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

---

## 2026-08-20 — ★ 选择头寸实测:**+26.4 点 / +1.46 token**,path selection 通过判据

### 结果

训练侧探针跑在 `ep5p0-ropefix`(`ckpt_faithful_ep_20260804_165215`,续训至 epoch 6,`LR=0`),
日志 `faithful_ep_20260820_011244.log`。取 12 个完整步的均值:

| 位置 | recall@1(= `position_k_acc`) | recall@16 | 缺口 |
|---|---:|---:|---:|
| pos0 | 0.858 | 0.987 | +12.9 点 |
| pos1 | 0.772 | 0.957 | +18.4 点 |
| pos2 | 0.697 | 0.915 | +21.8 点 |
| pos3 | 0.635 | 0.884 | +24.9 点 |
| **pos4** | **0.581** | **0.846** | **+26.4 点** |

```
hard_accept_len 3.845  →  oracle_accept_len_16 5.309    +1.46 token
```

**判据(8-19 记录里预先钉死:pos≥4 缺口 ≥10 点为真、<3 点结案)** —— 实测 +26.4,超线两倍半。
含义:末位置上草稿把正确 token 排进前 16 名的比例是 84.6%,而 argmax 只选中 58.1%;
**中间 26 个点是纯选择损失,与模型知不知道无关。**

缺口随位置单调放大(12.9 → 26.4),落点与 `num_spec=7` 分析指认的尾部一致,但**成因不同**:
块宽不匹配是"没训过",选择损失是"训过了但没选中"。二者正交,可叠加修。

### ★ 决定 selector 成本的发现:k=4 就够

末位(pos4)缺口的名次分布:

| k | 拿回 | 占全部缺口 |
|---:|---:|---:|
| 2 | +8.9 点 | 34% |
| **4** | **+15.9 点** | **60%** |
| 8 | +21.5 点 | 82% |
| 16 | +26.4 点 | 100% |

这改写了此前的成本担忧(129,280 全词表 topk × K 个位置的 host 侧开销):**按 k=4 算比 k=16
小一个量级,收益只少 40%**。第一版实现从 k=4 起步。

### 限定

1. **上界。** teacher forcing:`dspark/core.py:159` 喂给 Markov 头的是真前驱 token,服务端喂的
   是草稿自己的选择。下界须由服务侧 `DSPARK_TOPK_DUMP` 的首错 recall 给出。
2. oracle 假设"选择器每次都选对",真实 selector 只能拿回其中一部分。
3. 测的是训练分布(`arrow_0730_77w_dedup`),不是 gsm8k。

⟹ 结论的正确表述是**"值得往下做"**,而非"能拿到 +1.46 token"。

### 途中踩到的坑(全部是自己造的,记下来免得重犯)

★ **一个 bash 续行 bug 制造了两个假故障。** 给启动器加 `export` 时插在了
`nohup env \` 到 `torchrun` 之间——那是**一条命令**,注释吃掉逻辑行剩余部分后,
`env` 变成"没有命令"(只打印环境后退出,留下 `nohup.out`),`torchrun` 则脱离 env 前缀独立运行。
命令行上传入的变量靠 shell 继承仍在(EP 显示正常),**掩盖了故障**;丢的是只存在于启动器内部的:

| 丢掉的 | 表现 | 我当时的误判 |
|---|---|---|
| `DSPARK_HS_DUMP=1` | 每样本 `Response missing kv_transfer_params` | 一度怀疑服务侧起错了 build |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | `MAX_ANCHORS=512` 首步 OOM | 一度归因"续训导致碎片",**错的** |

已在 `nohup env` 上方加护栏注释。**教训:掩盖故障的不是缺失本身,而是"部分变量仍在生效"
造成的正常表象。**

另有两处误判也一并撤回:
- 「禁用融合 loss 导致 OOM」—— 错。Correction 那次同样带 `SPECULATORS_DISABLE_FUSED_LOSS=1`
  且 `MAX_ANCHORS=512`、还多背 97M 参数,连跑 34 小时无 OOM。
- 「融合核在昇腾上跑不了需要再验」—— 无需再验,port 分支已用实跑证明禁用它是安全的。

### 工具

`analyze_train_run.py` 新增 **SELECTION HEADROOM** 段:逐位置 recall@1 对 recall@2/4/8/16、
`hard` 对各 `oracle_accept_len_K`、末位名次分布、以及按预设阈值自动判读。普通训练日志没有这些键
⟹ 整段静默。

### 下一步

**零参数 Viterbi** —— 用**现有的** Markov 链做联合解码替代逐步 argmax,不加权重、不训练。
详见下一条记录。

---

## 2026-08-20 — 零参数 Viterbi:已实现并证明正确,**待机器**;附一处目标函数错配的预警

### 这个算法是哪来的(免得日后被当成"复现")

| 来源 | 内容 |
|---|---|
| DFlash2 博文**给了** | 打分公式 `S_t(a,b) = U_t(b) + ⟨A(a)⊙H(h_t), B(b)⟩`(逐字) |
| 博文**没给** | DP 递推、伪代码、selector 训练方法、2.0M 参数分配、T>0 拒绝采样 |
| 博文**实际的解码** | *"starting from the last verified token, **greedy follows the best successor** at each step"* —— **格上的贪心前向**,不是 Viterbi |
| **本工作加的** | ① 观察到我们的 Markov 头**本已构成同一条一阶链**(读代码得出,非博文所述);② 既然分数可分解为相邻项,用 **Viterbi** 求全局最优——**教科书算法(HMM 解码),既非本工作发明,也非 DFlash2 所用** |

⟹ 本实现的解码**强于**博文(全局最优 vs 贪心后继),代价仅是 DP 内 K×K 而非 K。
再次强调分支名 `dflash2-reproduce` 名不副实:这里没有任何"复现"。

### 实现

`vllm-ascend` `dflash2-reproduce` 分支,`DSPARK_VITERBI_K`(未设/0/1 → 逐字节回落原路径)。

- 候选取自一元项 top-k(与博文一致)⟹ 候选格与所评路径无关;
- t=0 前驱是已验证锚点(唯一已知 token),等价于今天的第一次迭代;
- 转移项复用模型自己的 `markov_bias()` ⟹ TP 下天然正确(`ParallelLMHead` 分片词表、
  `logits_processor` 负责 gather),代价是 materialize 全词表再取 K 列。若时延成瓶颈,
  精确优化是直接 gather `markov_w2.weight` 的 K 行(`O(K·K·rank)` 而非 `O(K·vocab)`),
  需 TP 感知的行索引;
- 仅贪心路径;形状恒定,且 `_sample_sequential` 本就在 ACLGraph 之外。

**正确性**:与穷举 `K^T = 4^5 = 1024` 条路径的最优解**逐 token 一致**(3/3 请求),
且 3/3 请求上与逐步贪心给出**不同**序列(否则改动等于没做)。

### ★★ 预警:目标函数错配 —— Viterbi 有可能打不过贪心

服务端是**前缀接受**:某位置一旦失配,其后全丢。真正要最大化的是

```
E[接受长度] = Σ_t P(位置 0..t 全对)
```

而 Viterbi 最大化的是**整块链分之和**。二者不等价:一条"牺牲 pos0、但 pos3/4 极好"的路径
链分可能更高而**实际收益为零**。前缀语义天然使早位置权重更大,而**贪心前向恰好偏向早位置**。

⟹ **若 Viterbi 持平甚至劣于贪心,这不是 bug,而是目标函数错配。** 该结论在跑之前先写下,
免得事后把它读成"联合解码没用"。

正因如此,该做的是三变体对照而非只上 Viterbi:

| 变体 | 最大化的量 | 意义 |
|---|---|---|
| **A. 贪心后继** | 逐步最优 | 忠实复刻博文解码,隔离"联合候选"这一半 |
| **B. Viterbi**(已实现) | 整块链分和 | 可能被前缀语义反噬 |
| **C. 位置加权 Viterbi** | 早位置加权(对齐训练损失的 γ=4 指数衰减) | 三者中最贴合 `E[接受长度]` |

A 与 C 尚未实现。

### 结果的天花板(先说清,免得高估)

Viterbi 只最大化**模型自己的**链分,只能拿回 8-20 实测那 +26.4 点 / +1.46 token 中的一部分;
而 DFlash2 的另一半——学出来的上下文门控 `H(h_t)`——按构造**不在这里**。
**因此持平也是有价值的结果:那恰好把价值定位到 `H` 上**,而这正是本实验存在的意义。

### 阻塞

**无 eval 机器。** 本实验是纯服务侧的,必须有一台带草稿的推理服务才能测。
机器到位后:起服务加 `DSPARK_VITERBI_K=4`(看到 `>>> [DSPARK_VITERBI]` 横幅即生效),
用同一套 `run_dspark_eval.sh` 对比 ledger 的 `ep5p0-ropefix ns5 = 4.4162`。
建议 K=4 起步(8-20 实测 k=4 已拿回 60% 缺口,成本比 k=16 小一个量级),绿了再看 K=8/16 的边际。

---

## 5. 解码器离线消融:k=4 首测 —— 联合解码为负,且发现一处实现错误(2026-08-20)

`decoder_ablation_probe.py`,`DECODER_K=4`,run `faithful_ep_20260820_042143`。

**保真门先过**:重放的 `today` 3.932 vs 该 run 自己的 `hard_accept_len` 3.946,差 0.014。
重放路径与服务端 `_sample_sequential` 一致,以下数字可信。

| 解码器 | accept_len | 相对 today | 赢 | 输 |
|---|---:|---:|---:|---:|
| today(服务端现行) | 3.932 | — | — | — |
| restrict(top-4 剪枝对照) | 3.428 | −0.496 | 2.2% | 26.2% |
| viterbi | 3.262 | −0.671 | 5.8% | 34.9% |
| decay(γ=4) | 3.333 | −0.601 | 5.7% | 32.5% |

拆开看:

```
剪枝成本 = today − restrict = 0.496
联合解码自身 = gain + 剪枝成本 = −0.175 (viterbi) / −0.105 (decay)
```

### 两个读数

**① `decay` 优于 `viterbi`(−0.105 vs −0.175)** —— 与跑之前登记的预测方向一致:
压低后位权重、靠近前缀接受的目标,伤害就小。**目标函数错配的诊断被数据支持。**

**② 但链分把未归一的 logits 跨位置相加,这是实现错误。**
`base_logits` 未归一,跨位置求和等于按各位置的 logit 量纲加权 —— 哪个位置 logit
张得开,哪个位置就主宰整条路径,这是任意的。序列解码求和的应当是 log-prob
(beam search 即如此),那样链分才是 log P(路径),Viterbi 最大化的才是"整块全对"的概率。
**逐位置 argmax 对归一化不变,所以 `today` 不受影响;联合解码受影响。**
⟹ −0.175 究竟是目标错配还是量纲伪影,k=4 这一轮分不出来。

### 已做的修改

探针改为一次跑六个解码器:`viterbiN` / `decayN` = 同样的位置权重但**每位置先 log_softmax**。
候选格(unary 值、id、全部 pairwise 转移)只构建一次并共享,所以多两个变体只多两趟 DP,
不再多扫一遍词表。四个 DP 变体均已与 4⁵=1024 条路径的暴力枚举逐 token 对齐;
`today` 与服务端规则逐 token 对齐;`restrict ≤ today` 断言通过。

分析器同步扩到六档,并加了一段"按位归一的效果"对照(`viterbiN − viterbi`、`decayN − decay`),
把这两种解释直接分开报。

### 下一步

同时抬两个量重跑一轮:`DECODER_K=16`(把 0.496 的剪枝伪影压下去)+ 六解码器。

```bash
# 与 k=4 那轮(faithful_ep_20260820_042143)逐字段相同,只改 DECODER_K
DSPARK_EP=1 BF16_EXPERTS=1 RECOMPUTE=1 COMPILE=0 \
TRAIN_PY=examples/ascend_npu_dflash/decoder_ablation_probe.py \
DECODER_K=16 DECODER_GAMMA=4.0 \
LR=0 EPOCHS=6 CKPT_FREQ=99 MAX_ANCHORS=128 \
SAVE_PATH=$RUN/ckpt_faithful_ep_20260804_165215 \
  bash examples/ascend_npu_dflash/train_dsv4_dspark.sh faithful
# 上一轮的实际取值以日志开头的横幅为准:head -20 $RUN/faithful_ep_20260820_042143.log
```

判读顺序固定为:保真门 → 剪枝成本 → 归一化差值 → 各档 gain。
- 若 `decayN` 转正 ⟹ 之前的负值是量纲伪影,联合解码本身可用,再谈服务端落地。
- 若归一化差值 ≈ 0 且各档仍为负 ⟹ 前缀接受与链分的错配是真的,**这条线判负**,
  头寸(pos4 +25.9pt)得靠训练侧或别的选择器去拿,不是靠换解码规则。

---

## 6. 读到 DFlash2 的官方实现:vllm-ascend #14533(2026-08-20)

`[Feature] add DFlash2 for MRV1`,作者 chenaoxuan,2026-08-19 开,777+/1−,9 文件。
跟的是上游 vllm#52816(未合)。权重已公开:HF collection `z-lab/dflash-2`。

### 打分式子:与我们从博文推的一致

`vllm_ascend/models/qwen3_dflash2.py::_score_edges` 展开:

```
S[l, p, c] = U_l(c) + Σ_r  A(a_p)[r] · H(h_l)[r] · B(b_c)[r]
           = U_t(c) + ⟨ A(a) ⊙ H(h_t) , B(c) ⟩
```

`predecessor_codebook` = A,`successor_codebook` = B,`hidden_projection` = H。

### ★★ 我们的 Markov 头就是它的两个码本

| 它 | 我们已有 |
|---|---|
| `predecessor_codebook` [V, rank] | **`markov_w1`** [129280, 256] |
| `successor_codebook` [V, rank] | **`markov_w2.weight`** [129280, 256] |
| `hidden_projection` Linear(hidden→rank) | **缺** |

缺口 = `Linear(4096 → 256)` = **1.05M 参数**。也对上了它 Table 2 的 +77.8M = 2×151936×256。

### ★ 关键纠正:我们的 `gated` 不是它的超集,而是弱在要害

| | H(h) | 值域 |
|---|---|---|
| 它 | `Linear(hidden → rank)` | **无界、带符号** |
| 我们 `gated` | `σ(Linear([h ; A(a)]))` | **(0,1),只能衰减,符号不变** |

sigmoid 门只能把某维关小,**永不放大、永不翻号** ⟹ gated bias 永远是 vanilla bias 的收缩版本。
对"rank-256 会不会不堪重负"这个问题,这是决定性的差别:

- sigmoid 门:只能从无条件 bigram 里做**减法**,256 维被蚕食 —— 担忧成立;
- 线性 H:每维自由带符号缩放,每个上下文对应一个**真正不同**的 rank-256 矩阵,不是掩码子集。

⟹ **不要用 `--markov-head-type gated`。** 坏的是 sigmoid,不是"让 Markov 头兼职 select"这件事。

### ⚠ 它的性能表没有隔离 selector

| Method | Spec Num | Mean AL | Acc-Rate per Pos(%) | Throughput(req/s) |
|---|---|---|---|---|
| No-Spec | — | — | — | 0.21 |
| MTP | 8 | 6.17 | 94, 86, 77, 68, 59, 51, 44, 38 | 0.92 |
| DFlash2 | 8 | **6.73** | 94, 87, 80, 74, 68, 62, 57, 51 | **1.02** |

(Qwen3.8-27B,DP1/TP2,gsm8k 前 300,batch 16,temp 0,A2B3-NPU,eager)

对照组是 **MTP(自回归逐 token)**,而 DFlash2 是**新骨干 + selector 的整包**。
`_grouped_conv`(块内输入相关的分组因果卷积)是**草稿骨干**组件,不是 selector ——
PR 描述把两者混为一谈。**⟹ 那 +0.56 不能记在 selection 上。**
外部没有任何数字说明 selection 单独值多少;§2 我们自己量的头寸仍是唯一证据。

### ✓ 交叉验证:它的解码器也是贪心,不是 Viterbi

`dflash2_greedy_selector_walk_kernel`:`prev_idx=0` 起步,逐步取 argmax(平局取小下标)。
我们 §5 的消融独立测出联合解码/Viterbi 在前缀接受下为负(−0.175),`decay` 优于 `viterbi`。
**两条独立路径同一结论**,给我们的消融结果加分。

### ★ 我们的服务端不需要它的 top-k 走链

它算 `[B, L, top_k, top_k]` 的整表再用 Triton 核走 —— 那是**核效率**选择,不是算法需要
(贪心每步只访问一个前驱)。我们的 `_sample_sequential` 本来就是逐步全词表 argmax,
**算法上等价或更强**(全词表 ⊃ top-k;§5 已测 top-k 剪枝在 k=4 要花 0.496 token)。
⟹ 服务端改动只是"`bias()` 多收一个 hidden、多一个 Linear",约 10 行,不需要 Triton 核,
也不需要 top-k。比原先估的 30 行还小。

### 对计划的影响

1. **从零训**:加强。A/B 码本必须与 H **共同学出**;续训等于把 H 螺丝钉在一组从没见过调制的基底上。
2. **谱检查(§7 脚本)照跑**,但问题变了:不是"要不要做",而是"**256 维够不够同时扛无条件 bigram + 上下文调制**"。饱和 ⟹ 新头用更大 rank,不是放弃。
3. **实现它的形式,不用 `gated`** —— 新 head type `dflash2`。
4. 服务端有模板,且就在 vllm-ascend 内。

---

## 7. 曝光偏差实测 0.014;谱检查;定案用**加性** SelectHead(2026-08-20)

### 7.1 ★ 曝光偏差 = 0.014 token —— §5 的保真门顺带量出来了

|  | 前驱来自 | accept_len |
|---|---|---:|
| 训练 `hard_accept_len` | **真** token(`dspark/core.py:159` `prev_token_ids = block_tokens`) | 3.946 |
| 重放 `today` | **草稿自己的选择**(`prev = pick` 逐步喂回) | 3.932 |

写重放时"喂自己的选择"是当保真度要求做的,没意识到它同时构成了对照。
**训推曝光偏差 = 0.014 token,在噪声带内。**

**而且这是结构决定的,不是运气**:`accept_len` 是**前缀**度量,第一个挑错的位置就把整块截断,
所以"前驱是错的"只发生在**已被丢弃的位置**上 —— 前缀接受让曝光偏差自我封顶。
`num_spec` 变大也不会线性放大,同样被截断吃掉。

**★ 推论:上下文调制不新增任何暴露面。** `H(h_t)` 的输入 `h_t` 来自并行骨干的**一次**前向,
在任何挑选发生**之前**就算完(`proposer.py:923`,`base_logits` 上一行),训练与服务完全一致;
另一个输入就是 vanilla 已经在用的同一个 `prev`。
⟹ 早先"更强的 select 头会因 teacher forcing 而更危险"的担心**撤回**,有实测也有机制。

### 7.2 谱检查:`ep5p0-ropefix` 的 Markov 基底

`markov_spectrum_probe.py`,`ckpt_faithful_ep_20260804_165215/4`:

```
w1 (前驱嵌入 A)     有效秩 255.0 / 256  (99.6%)   能量覆盖 50%->123  80%->202  90%->229  95%->242  99%->254
w2 (后继嵌入 B)     有效秩 142.7 / 256  (55.7%)   能量覆盖 50%-> 93  80%->186  90%->220  95%->238  99%->253
★ 转移矩阵 w2·w1ᵀ   有效秩 110.0 / 256  (43.0%)   能量覆盖 50%-> 89  80%->183  90%->218  95%->237  99%->252
```

⚠ **脚本当时的判词"基底没被吃满"过于乐观,以此处为准:**
复合矩阵的 participation ratio 是 110(看着有余量),但**能量尾巴铺满 256 维**(99% 要 252),
且 **A 本身 99.6% 近乎满秩**。所以不是"有 146 维闲着",而是"头部集中、但没有一块真正空着"。
**混合信号,不构成"放心复用同一组基底"的结论。**

### 7.3 定案:**加性** SelectHead,不用融合式 `dflash2` 头

```
S_t(a,b) = U_t(b) + ⟨A(a), B(b)⟩ + ⟨A′(a) ⊙ H′(h_t), B′(b)⟩
              无条件 bigram(不变)      上下文选择项(新增)
```

两种实现**数学上等价**(同权重下 bias 逐比特相同,已验证);选加性是**工程属性**上的判断:

1. **优雅退化。** 融合式若转换器漏掉 `H`,服务端会拿"在有 H 前提下训出的 w1/w2"算无 H 的
   bias —— **静默错误**,正是 Correction 头那次的死法。加性项漏掉 = 退回今天的行为,**无增益但无错**。
2. **它本身就是消融。** 把该项置零 = 今天的模型逐比特相同 ⟹ "selection 单独值多少"直接可测。
   融合式永远拆不开(两臂的 w1/w2 本身就不同)。
3. **服务端可先落地再等权重** —— 训练前恒为 0,那 10 行可以先合、先验证是 no-op。

代价:`A′+B′` = 2×129280×256 = **66.2 M**,`H′` = 1.05 M。占 21B 草稿的 0.3%;
服务端每步多一次 `[256 × 129280]` matmul,对 1.5B 激活的骨干前向可忽略。

初始化 `B′ = 0` / `H′ = 1` ⟹ 整项 step 0 恒为 0,配对 A/B 从第一步就干净。
B′ 先动、A′ 随后解冻(已验证),零初始化输出的标准形态(同 LoRA、同我们的 `--dflash-*`)。

`markov_head_type="dflash2"`(融合式,忠实于 #14533)保留在树里,以后当 A/B 的另一臂。

### 7.4 本轮启动命令(BEST 基线 + 本次 SELECT 臂)

与产出 `ep5p0-ropefix` 的 canonical 命令**逐字段相同,只多 `SELECT_RANK=256`**
⟹ 同种子同数据,batch 序列逐步一致,`faithful_ep_20260804_165215.log` 即免费配对基线。

```bash
# 前置:115/116 的 HS_DUMP=1 serve 必须先起着(在线取隐状态)
cd /home/a00652497/dspark_austin/speculators
git checkout dflash2-reproduce && git pull

DSPARK_EP=1 BF16_EXPERTS=1 RECOMPUTE=1 COMPILE=0 \
DSPARK_MOE_BALANCE=1 DSPARK_MOE_BALANCE_RATE=1e-3 DSPARK_LOG_EXPERT_LOAD=1 \
INIT_LAYER=1 INIT_MOE_NO_ROUTER=1 \
SELECT_RANK=256 \
LR=2e-4 EPOCHS=5 MAX_ANCHORS=512 CKPT_FREQ=0.5 \
DATA=/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow_0730_77w_dedup \
  bash examples/ascend_npu_dflash/train_dsv4_dspark.sh faithful
```

启动后走开之前**必须确认两件事**:
- 横幅里有 `select_rank=256` —— 没有就是旗标没透传,白跑 24 小时;
- 前几十步 `hard_accept_len` 与基线同量级 —— 零初始化生效的话起步就是恒等,崩了说明没生效。

读数:

```bash
RUN=/home/a00652497/dspark_austin/run
python3 examples/ascend_npu_dflash/analyze_train_run.py \
  $RUN/faithful_ep_<新TS>.log --baseline $RUN/faithful_ep_20260804_165215.log \
  --label SELECT --baseline-label ROPEFIX --out $RUN/cmp_select
```

⚠ **读数的不对称**:`A′/B′` 是全新的 129280×256 嵌入表,1 epoch 内还年轻。
**正结果决定性;打平只能说"这个预算内没看到",不能判死。**
(参照:Correction 头 17k 步就有 +0.038,说明新模块在这套设置里确实会早早出信号。)

---

## 8. 机制说明:从 Markov 头到 select 头,以及**走链的访存成本**(2026-08-20)

写给以后要跟别人解释这件事的人。符号先立起来。

### 8.1 符号

| 记号 | 含义 | 实际取值 |
|---|---|---|
| `V` | 词表 | **129,280** |
| `d` | 草稿隐藏维 | **4096** |
| `r` | 码本秩(`markov_rank` / `select_rank`) | **256** |
| `K` | 块宽(= `num_speculative_tokens`) | **5** |
| `h_t ∈ ℝ^d` | 块内位置 `t` 的草稿隐状态 | |
| `a` | **前驱** token id(位置 `t−1` 实际选中的;`t=0` 为 anchor) | |
| `b` | 位置 `t` 的候选**后继** token id | |
| `⊙` | 逐元素乘 | |

| 记号 | 代码 | 形状 | 读法 |
|---|---|---|---|
| `A` | `markov_w1.weight` | `[V, r]` | 第 `a` 行 = "该词作为**前驱**"的编码 |
| `B` | `markov_w2.weight` | `[V, r]` | 第 `b` 行 = "该词作为**后继**"的编码 |
| `A′,B′` | `select_w1/w2` | `[V, r]` | 同上,属 select 头 |
| `H′` | `select_hidden` | `Linear(d→r)` | 把隐状态投进**同一个 r 维空间** |

### 8.2 块草稿的结构性弱点

一次前向同时产出 `h_0…h_{K−1}`,`U_t(b) = ⟨W_lm[b], h_t⟩`。
**`U_t` 只是 `h_t` 的函数,而所有 `h` 在任何 token 被选中之前就算完了** ——
整块是 K 个互相独立的边缘分布,`U_4` 不知道位置 3 选了什么。

### 8.3 Markov 头 = 一个**全局**转移矩阵的低秩分解

```
bias_M(a)[b] = ⟨A(a), B(b)⟩         ⟺   M = A·Bᵀ ∈ ℝ^{V×V},   M[a,b] = 见 a 之后 b 加多少分
```

`V²=1.67e10` 压进 `2Vr=6.6e7`,约 250× 压缩。服务端 `_sample_sequential` 逐步用**实际选中的**
前驱查 `M` 的一行,把 K 个独立边缘串成一条链。

**上限**:`M` 只有一个,对所有上下文相同。`M["New","York"]` 只能是全语料平均值,
可旅游稿要 York、地理要 Zealand、产品页要 features —— 只能平均掉。

### 8.4 select 头 = 让转移矩阵随上下文变

```
bias_S(a,h_t)[b] = ⟨A′(a) ⊙ H′(h_t), B′(b)⟩     ⟺   M(h_t) = A′·diag(H′(h_t))·B′ᵀ
```

把 `r=256` 理解成 **256 种转移模式**;`H′(h_t)_k` 是第 k 种模式在当前上下文下的开关兼增益:
`>1` 放大、`≈0` 关闭、`<0` **反转**。(这就是 `gated` 不行的原因:`σ(·)∈(0,1)` 只能关小。)

### 8.5 三项 = 主效应 + 交互项

```
S_t(a,b) =  U_t(b)                    只依赖 h_t         主效应①
         +  ⟨A(a),        B(b)⟩        只依赖 a           主效应②
         +  ⟨A′(a)⊙H′(h_t), B′(b)⟩     同时依赖两者  ★ 交互项
```

★ **没有新信息进来**(`U_t` 本就是 `h_t` 的读出),新的是**函数形式** ——
`h_t` 与 `a` 的双线性耦合,前两项的形状怎么都写不出来。
只有主效应的模型在结构上无法表达"a 的作用随上下文而变"。

### 8.6 服务端循环(全标注)

```python
prev = anchor                                  # 目标模型上一轮验证通过的最后一个 token
for t in range(K):
    s  = U_t                                   # [V]  块并行前向的一元项
    s += B  @ A[prev]                          # [V]  全局 bigram        bias_M
    s += B' @ (A'[prev] * H_proj(h_t))         # [V]  上下文相关转移      bias_S ★新增
    token_t = s.argmax()
    prev = token_t
```

### 8.7 ★ 走链的访存成本 —— 不可忽略,且我先前讲轻了

一个码本 `[V,r]` bf16 = **66.2 MB**。这个循环**每个位置各读一遍整张表**:

| | 每个投机步的权重读 |
|---|---:|
| 今天(只有 Markov) | K × 66.2 = **331 MB** |
| 加 select 之后 | K × 132.4 = **662 MB** ← 翻倍 |

算力只有 0.66 GFLOP,**纯访存瓶颈**。折算到实测步时(一个投机步 71.39 ms → 4.831 token):

| 码本 GEMV 达到 | 现有走链 | 加 select | 增量 | **收支平衡需 Δaccept ≥** |
|---|---:|---:|---:|---:|
| 84 GB/s | 3.94 ms | 7.88 ms | +3.94 | **+0.267** |
| 200 GB/s | 1.65 ms | 3.31 ms | +1.65 | **+0.112** |
| 400 GB/s | 0.83 ms | 1.65 ms | +0.83 | **+0.056** |
| 800 GB/s | 0.41 ms | 0.83 ms | +0.41 | **+0.028** |

⟹ **select 项必须买到 0.03~0.27 token 才回本**,具体落点取决于码本 GEMV 的实测带宽 —— 这个数
我们还没量过,**上服务端之前必须量**。

### ⚠ 但以上只在 **conc1** 成立 —— 并发会把它摊掉

码本读取是**每步固定**的,与 batch 无关:一步里 R 个请求同处位置 `t`,`markov_bias` 是一次
`[V,r] × [r,R]` 的 **GEMM**,码本读一遍,R 个请求分摊。

| 并发 R | 走链码本流量/步 | **每 token 分摊** |
|---:|---:|---:|
| 1 | 662 MB | **137 MB** |
| 4 | 662 MB | 34 MB |
| 16 | 662 MB | 8.6 MB |
| 48 | 662 MB | **2.9 MB** |

⟹ 上表那个"回本门槛"**只对 batch-1 解码有效**。conc1 恰好是我们对外报 2.27× 的那个点,
所以它对**那个数字**仍然要紧;但它**不是上线的阻碍** —— 真实服务并发下这项趋近于零。

### 训练侧完全没有这个问题

teacher forcing ⟹ K 个位置的前驱全是已知的真 token,一次 `[N·K, r] @ [r, V]` 的 GEMM 算完,
**码本读 1 遍,不是 K 遍**。参数、显存、FLOPs 三项上,select 与 markov 等量,都不重。
那个逐步循环**只存在于推理**,因为 `prev` 依赖上一步的 argmax,链无法并行。

### 8.8 ⚠ 纠正:DFlash2 的 top-k 不只是核效率,是**算法级降本**

§6 我写过"它的 `[B,L,k,k]` 整表是核效率选择,我们不需要"。**成本这一面我漏了。**

| | 每位置访存 | 每步算力 |
|---|---|---|
| 全词表打分(我们现在) | **132 MB**(整张 A′/B′) | 331 M MAC |
| top-k 打分(k=16) | **16.4 KB**(只 gather k 行) | 0.33 M MAC |

`O(K·V·r) → O(K·k²·r)`,访存差 **~4 个数量级**,走链成本基本归零。
代价是候选被剪到 k —— §5 实测 **k=4 要花 0.496 token,k=16 未测**。

⟹ 所以"重"**不来自"有 Markov 头",来自全词表 argmax 这个解码方式** ——
是可改的解码选择,不是架构差异。

★ 另一处设计差别:**DFlash2 只有一对码本**,式子里没有独立的 `⟨A(a),B(b)⟩` 主效应。
因为若 `H′(h)` 学成常数向量 `c`,`⟨A(a)⊙c, B(b)⟩` 即退化为一个固定低秩转移矩阵 ——
**主效应是交互项的特例**,他们的形式更简约。我们背两对码本是为了 §7.3 那三条工程属性
(优雅退化 / 结构性消融 / 服务端可先落地),代价是服务端 2×;走 top-k 后这点差别也消失。

⟹ 两条路都成立,**由 `sel_gain` 的大小决定走哪条**:

- `sel_gain` 明显大于 8.7 表的门槛 ⟹ 全词表版直接上,简单;
- `sel_gain` 处在边缘 ⟹ 改 top-k 版把成本抹掉,再用 `DECODER_K=16` 那轮量剪枝代价,两边一减看净值;
- `sel_gain` ≈ 0 ⟹ 都不上。

**待办**:① 量码本 GEMV 的实测带宽(服务端 profile);② 跑 `DECODER_K=16` 定 top-k 的剪枝代价。
两个都不需要训练,都可以在等这轮训练的 24 小时里做。
