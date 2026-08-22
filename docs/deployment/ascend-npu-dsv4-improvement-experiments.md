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
| **训练权重** | `run/ckpt_faithful_ep_20260818_122129/0`(1.0 epoch)与 `/1`(1.5 epoch)。标准 HF 格式:`config.json` + `model.safetensors`(2026-08-21 在盒子上核实;先前记为「EP 分片 DCP」是错的)。仍需 `convert_dspark_to_vllm.py` 做**键名重排** `layers.* → mtp.*`,但见下方 ⚠ |
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

---

## 9. 判据的校准:用 Correction 当标尺,当场证伪了"等效领先步数"(2026-08-21)

### 9.1 起因

SELECT 早期领先(step 1656 时 Δ +0.086 vs Correction +0.063),但**早期领先什么都不能说明** ——
Correction 在前 2000 步的领先也一路上涨(−22 → 320),最终却收敛到 +0.017。
于是造了个判据想区分"真提升"与"只是早到",并**拿结局已知的 Correction 去校准**。

### 9.2 判据一(已废弃):等效领先步数 = 反演基线曲线

想法:"基线要跑到第几步才达到这个臂现在的水平"。纯横向提前 ⟹ 恒定;水平抬升 ⟹ 增长。

**在 Correction 上直接判错 —— 判成"水平抬升,值得走到 eval"。** 两重结构性缺陷:

1. **在唯一能分辨的区间自我失明。** 领先 ∝ 1/斜率,曲线越平误差带越宽;
   Correction **8000 步之后的点全被误差带排除**,而答案恰恰在那里。
2. **平的 delta 会被读成增长。** ≤5000 的那几个点 delta 其实平在 +0.06 上下,
   领先之所以从 −22 涨到 1348,只是基线斜率在变小、同样的 delta 除以更小的数。

（另有三个实现 bug 是靠合成臂发现的:未跑到该步数的臂仍被比较、两边平滑变换不一致导致
正 delta 报负领先、以及"基线与自己比"被判成水平抬升。修复见 `d511f6f`。）

### 9.3 判据二(现行):直接看 **delta** 的走势

delta 没有除法,不会爆炸,而且直接就是我们想问的:

```
收敛到同一渐近线  ⟺  delta → 0        （纯横向提前也是这个形状）
渐近线不同        ⟺  delta → 非零常数,其值即为渐近线之差
```

合成臂验证(真值已知):独立同分布采样、**纯提前 300 步**两者都读出"同一渐近线",
只有"渐近线抬高 3%"那个读出抬升。

### 9.4 ★ Correction 的标定值

```
delta:  2000:+0.062  3000:+0.054  5000:+0.066  8000:+0.034
       12000:+0.028 17000:+0.023 24000:+0.024 32000:+0.024 38900:+0.023
```

**从 +0.062 衰减到 +0.023 的平台(占当时水平的 0.6%),17000 步之后不再变。**
比零多一点点,但小到不值得 —— 与当初"两条曲线收敛到同一渐近线"的定性判断一致,现在有了数值。

⟹ **以后任何改动的训练侧 delta,拿 +0.023 当"约等于没有"的刻度。**

### 9.5 用法

```bash
python3 examples/ascend_npu_dflash/analyze_train_run.py \
  --baseline $RUN/faithful_ep_20260804_165215.log \
  --arm SELECT=$RUN/<新TS>.log \
  --arm CORRECTION=$RUN/faithful_ep_20260818_122129.log \
  --at 500 1000 2000 5000 10000 20000
```

看 **★ delta 列**与末尾判读;`lead(steps)` 列保留为参考,曲线变平后标记为不可用。
⚠ 训练侧 SOFT accept_len 仍只是代理,裁决只能是转换后的服务端评测。

### 9.6 ★ 预注册:决胜步数与有效性门槛(2026-08-21,**在看到 SELECT 结果之前写下**)

**决胜 step = 17000。** 依据是 Correction 的衰减进度(2000 步为起点,17000+ 的平台为终点):

| step | delta | 衰减完成 |
|---:|---:|---:|
| 2000 | +0.062 | 0% |
| **5000** | **+0.066** | **−10%** ⚠ 比起点还高 |
| 8000 | +0.034 | 73% |
| 12000 | +0.028 | 88% |
| **17000** | **+0.023** | **平台** |
| 38900 | +0.023 | 平台 |

⚠ **5000 步是陷阱**:那里 delta 比 2000 步还高,单点读数会给出假绿灯,而它正处在衰减前夜。
**必须看走平(末 3 个采样点的变化落在误差带内),不能看单点大小。**

自 run 启动起算(实测 3.26 s/step,含尖峰):5000→4.5h,8000→7.2h,12000→10.9h,**17000→15.4h**。

**有效性门槛**(以 Correction 平台 +0.023 = "约等于没有" 为刻度):

| SELECT 走平后的 delta | 占比 | 判定 |
|---|---|---|
| ≤ +0.03 | ≤0.8% | **死** |
| +0.03 ~ +0.08 | 1~2% | **边缘** —— 须先用 top-k 抹掉服务端成本 |
| **≥ +0.10** | **≥2.5%** | **做** —— 上 vllm-ascend,测真实接受长度 |
| ≥ +0.20 | ≥5% | 明确值得 |

`+0.10` 的来历:训练侧当前约 3.9,+0.10 = +2.6%;该相对幅度若带到服务端约 +0.11,
落在 §8 回本区间 (+0.03~0.27) 的中段。⚠ 训练侧 soft 与服务端 hard **无可靠换算**,量级估算而已。

**预测(写死待验)**:SELECT @2000 = +0.141,是 Correction @2000(+0.062)的 **2.27×**。
若走同样的衰减形状(×0.379),**平台应落在 +0.053** —— 即"边缘"区间。

⟹ 真要赢,SELECT 必须**衰减得比 Correction 慢**。三选一:

```
平台 明显 > 0.053  ⟹ 衰减更慢 ⟹ 真有东西,做服务端
平台 ≈ 0.053       ⟹ 同一种"早到",幅度大些 ⟹ 边缘,先做 top-k 降本再说
平台 ≤ 0.03        ⟹ 死
```

---

## 10. 上游化:侵入性预算与 `train/` 的拆分(2026-08-21)

维护者(`shanjiaz`,#952)的要求:拆成两份独立详细设计 —— ① DSV4-Flash DSpark 草稿模型定义、
② 专家并行训练;判据是**「侵入性不大就能合」**。于是先量侵入面。

⚠ 量之前先修一个坑:我们 fork 的 `main` 停在 2026-07-02 且是我们自己的提交,**脏了 7 周**。
必须对**真上游** `vllm-project/speculators` 的 `main` 取 merge-base(2026-07-17,其后 88 个上游提交),
否则统计里会混进大量上游自己的改动(peagle / eagle3 / dflash 那些都不是我们写的)。

### 10.1 净改动(排除 `examples/` `docs/` `.claude/` `.github/`)

| | 文件 | 增 | 删 | 性质 |
|---|---:|---:|---:|---|
| **A 新目录 `models/dsv4_dspark/`** | 15 | +2622 | **−0** | **纯新增,零删除** |
| B `models/dspark/` | 4 | +93 | −16 | 小 |
| **C ★ `train/`** | 6 | +510 | −85 | 真正的侵入面 |
| D ★ `scripts/` | 5 | +941 | −6 | 其中 657 行是三个全新脚本 |
| E `tests/` | 3 | +201 | −7 | |
| **F ★ 其他共享文件** | 6 | **+98** | **−3** | 几乎可忽略 |

F 类全部六处:`models/metrics.py` +53、`data_generation/preprocessing.py` +21、`model.py` +12、
`models/utils.py` +5、`models/dflash/core.py` +4、`models/__init__.py` +3(注册)。

### 10.2 `train/` 拆开:五类,其中两类与 EP/DSV4 都无关

| 类别 | ~行数 | 位置 |
|---|---:|---|
| **① EP 通用机制** | ~140 | `distributed.py`(`shard_experts_as_dtensor`、`apply_fully_sharded` 加 mesh/ignored)、`trainer.py`(mesh 构建、专家跳过 broadcast、`bf16_experts` AMP 选项) |
| **② 通用 QoL(与 EP/DSV4 无关)** | ~100 | `dataloader.py --no-validation`、`logger.py` 纯文本镜像、`trainer.py::_gpu_mem_stats` |
| **③ 在线 HS 管线** | ~160 | `data.py` 的 `_hs_fetch_session` / `_fetch_hs_remote` / `_dump_generate_hs` |
| ④ EP 诊断 | ~30 | `trainer.py` 的 align 屏障 + per-rank fetch 统计 |
| ⑤ 大模型 meta 初始化 | ~30 | `distributed.py::build_on_meta` |

### 10.3 ★★ `train/` 里对 DSV4 的硬耦合 = **3 行**

```
distributed.py:250   if type(module).__name__ != "GroupedExperts":
trainer.py:341-342   _ep_local = getattr(self.model, "ep_local_param_keys", None)
                     _expert_keys = set(_ep_local()) if callable(_ep_local) else set()
```

两处**都已是鸭子类型**(按类名字符串 / `getattr`+`callable` 兜底),不 import 我们的模型。
⟹ **EP 那块本质上已经模型无关**;设计文档的核心命题就是把这两处升格为正式约定
(一个 expert-module protocol,或显式传 predicate),侵入面便只剩
"给 `apply_fully_sharded` 加 `mesh=` 与 `ignored_params`"。

### 10.4 提交顺序建议

1. **② 那 ~100 行先单独发 PR** —— 与 DSV4、与 EP 都无关的纯可用性改进,零争议,先拿小 merge 建立信任。
2. **① 草稿模型定义**(纯新增、零删除)。建议把 `backbone/` 里的
   `moe_ep.py` / `moe_grouped_gemm.py` / `moe_compile.py`(+545)剥给 EP 那份,
   模型定义即为 **~2077 行纯建模代码**。代价:① 单独合入后上游**训不动**该模型,须在文中讲明。
3. **EP 训练**(①的另一半 + `train/` 的 ①④⑤)。
4. ③ 在线 HS 管线单独成篇,或留在 fork。

### 10.5 关于「训练完能否直接推理」——现状

- **上游别的模型不需要出站转换。** `src/speculators/convert/` 全是**入站**的
  (EAGLE3 / MTP / DFlash 的研究仓 checkpoint → speculators 格式);vLLM 直接读 speculators 格式。
- **我们的检查点也是标准 safetensors**(2026-08-21 核实:`config.json` + `model.safetensors`)。
  上游 `checkpointer.py::DistributedCheckpointer` 用
  `get_model_state_dict(full_state_dict=True, cpu_offload=True)` 合并后由 rank0 `save_pretrained` —
  把专家做成与 FSDP 其余部分同 mesh 的 `Shard(0)` DTensor,正是为了让这条路不用特判。
  **我们一行 `checkpointer.py` 都没改。**
- ⟹ **转换那一步的唯一理由是键名重排 `layers.* → mtp.*`** ——
  因为 vllm-ascend 的 DSV4-DSpark 加载器是照**官方发布草稿的 `mtp.*` 布局**写的。
  **这是推理侧的选择,与 EP 无关**;若 vllm-ascend 直接认 speculators 布局,这一步即可消失。

---

## 11. 上游 vllm-ascend / vLLM 现状,以及 `method: "mtp"` vs `"dspark"`(2026-08-21)

### 11.1 上游已原生支持 DSV4 DSpark,但只认发布布局

`vllm_ascend/models/deepseek_v4/dspark.py`(`DSparkDeepseekV4ForCausalLM`)读 `mtp.{i}.*`,
做的是 **DeepSeek 研究命名 → vLLM 命名**的映射(`.attn.→.self_attn.`、`.w1.→.gate_proj.` …)。
**`speculators` 出现 0 次。**

而 **vLLM 本体早有通用 speculators 注册表** `vllm/transformers_utils/configs/speculators/algos.py`:

```
eagle3 → Eagle3LlamaForCausalLM / Eagle3Qwen3ForCausalLM     ← 按目标模型分支
peagle → PeagleLlamaForCausalLM / PeagleQwen3ForCausalLM
dflash → DFlashDraftModel
dspark → Qwen3DSparkModel                                     ← 写死 Qwen3
```

`dspark` 那条原样透传的正是我们的字段:`markov_rank` / `markov_head_type` / `block_size` /
`enable_confidence_head` / `confidence_head_with_markov`。

⟹ **转换器现在仍需要**(§10.5),但补上它只有两处:
vLLM `algos.py` 加一条按目标模型的架构分支(`eagle3`/`peagle` 已有先例),
vllm-ascend `load_weights` 除 `mtp.*` 外也接受 speculators 布局。

### 11.2 ★ PR #12968 不是性能优化,是 MRV2 适配

`[Feature][MRV2] Support DSV4 Dspark`,wxsIcey,2026-08-20 合入,231+/31−,11 文件。
用**发布草稿**(w4a8)在 gsm8k/400 prompts/num_spec=5/temp0/eager:

| | mean accept_len | 吞吐 |
|---|---:|---:|
| mrv1 | 4.33 | 877.16 tok/s |
| mrv2 | 4.32 | **1191.79 tok/s** |

逐位置接受率几乎一致(0.8951/0.7780/… vs 0.8960/0.7719/…)⟹ **纯引擎侧收益,与草稿无关。**
diff 全是接线:非因果并行草稿的 `dspark_swa_indices`、`SupportsEagle3` 接口、
把 RoPE positions 转发进草稿 attn metadata 的上下文管理器、量化配置继承补丁。

⟹ **那 36% 是"换 ModelRunner"拿的,不是"改草稿"拿的**,对任何模型都成立。
我们停在 `pr-12006`,既无 MRV2 也无此 PR。**这条应列为与 select 线并行的独立待办**
—— 不需要训练,收益量级又高一档。
（注:他们那份发布草稿 gsm8k 4.33 vs 我们自测同一草稿 4.658,差别在他们目标模型是 w4a8 量化,不可直接比。)

### 11.3 `method: "mtp"` vs `"dspark"` —— **proposer 内相同,vLLM 内核不同**

`get_spec_decode_method`:

```python
elif method in ("mtp", "dspark") and getattr(...hf_config, "dspark_block_size", False):
    return AscendDeepSeekV4DSparkProposer(...)      # 两个别名同等
```

且 `AscendDeepSeekV4DSparkProposer(AscendDsparkProposer)` 在 `__init__:57` 无条件
`self.method = "dflash"` ⟹ `llm_base_proposer` 里所有 method 分支两者一致。

**但 vLLM 内核在 proposer 存在之前就已按字符串分叉**(`vllm/config/speculative.py`):

| | `"mtp"`(我们现在) | `"dspark"` |
|---|---|---|
| `use_eagle()` | True | True(相同) |
| `use_dspark()` | **False** | True |
| parallel drafting | 内核**不设** | 内核**自动开启** |
| `dspark_draft_topk` 校验 | 跳过 | 执行 |
| MTP 整除检查 `num_spec % n_predict` | **执行**(无意义) | 不执行 |

★ 并行草稿本该由内核按 method 打开,我们靠 ascend proposer 在 `__init__:58` 手动补
`self.parallel_drafting = True`。**能跑,但是在用下游补丁弥补上游未触发的配置。**

★ 顺带:**内核里已有 `dspark_draft_topk`** —— 即 §8.8 讨论的 top-k 草稿打分,上游可能已铺好路,值得单独看。

★ **上游缺口(适合一行 PR)**:自动识别写的是
`"dspark" in model_name.lower() or "Qwen3DSparkModel" in architectures`,
不覆盖 `DSparkDeepseekV4ForCausalLM` ⟹ 我们的草稿永远无法被自动识别。

### 11.4 待测变体

`examples/ascend_npu_dflash/serve_dsv4_a3_singlenode_specmethod.sh` —— 母本逐字节相同,
只把 `"method":"mtp"` 换成 `"${SPEC_METHOD:-dspark}"`。
⚠ **不是零风险改名**,它改变 vLLM 内核行为,必须实测。
验收:逐位置接受率与 `ep5p0-ropefix`(gsm8k 4.849)在噪声内一致,且 tok/s 不降。
115/116 现在在给训练供 HS,等空出来再跑。

---

## 12. 服务侧待办块 + SELECT 的 3000 步读数(2026-08-21)

### 12.1 SELECT @3000:两个极端夹住门槛,判不了

| step | SELECT Δ | CORRECTION Δ |
|---:|---:|---:|
| 500 | **+0.256** | −0.056 |
| 1000 | +0.169 | +0.006 |
| 1500 | +0.175 | +0.047 |
| 2000 | +0.137 | +0.062 |
| 2500 | +0.130 | +0.073 |
| **3000** | **+0.137** | +0.054 |

**两条曲线形状相反**:SELECT 从 +0.256 一路衰减(零初始化的 `B′` 从 step 1 就贡献);
Correction 从 −0.056 上升,**5000 才见顶 +0.066**,之后崩到 +0.023 平台。
⟹ **3000 步正是 Correction「看着还不错」的位置,陷阱区尚未穿过。**

```
① 若已到平台(近三点 0.137/0.130/0.137 噪声内持平)      ⟹ +0.137  →「做」
② 若照 Correction 自 3000 起的衰减(×0.435)             ⟹ +0.060  →「边缘」
```

§9.6 预注册门槛 **+0.10 正好夹在两者之间** ⟹ 现在无法判定,须等 17000。
(预注册预测 +0.053,按 3000 步重算 +0.060,吻合。)

⚠ 读数必须用 **delta 列**;`lead(steps)` 那套判据已于 §9.2 证伪,分析器 `f64bab9` 起改判。

### 12.2 ★ 待办块:等 115/116 空出来,一次起服务测三件

**顺序不可换** —— (c) 先验无损再谈提速,否则速度多少都没意义。

```
(a) method=dspark 验证                                              便宜,先做
    脚本  examples/ascend_npu_dflash/serve_dsv4_a3_singlenode_specmethod.sh
    验收  逐位置接受率与 ep5p0-ropefix(gsm8k 4.849)在噪声内一致,tok/s 不降
    收益  不再靠 ascend proposer 手动补 parallel_drafting(见 §11.3)

(b) MRV2 的 AR 基线 vs MRV1                                          ★ 决定性
    做法  同一目标模型、同数据集,分别在 MRV1 / MRV2 上测【无投机】吞吐
    读法  AR 也快 ~36% ⟹ 引擎级普涨:加速比 2.27× 不变,但 rollout 与 HS prefill 都受益
          只有投机臂快 ⟹ 加速比会涨(对外报数好看),但 HS 瓶颈不受益
    ⟹ 这一测决定「35% 的 HS 瓶颈能否靠换引擎解决」

(c) rollout 挂草稿:先 diff,再测速
    第一步  同一批 prompt 跑两遍(带 DRAFT / 不带),【逐行 diff】
    第二步  完全一致才测吞吐;不一致则整条路作废
```

### 12.3 为什么 rollout 能挂草稿、HS 转储不能

**rollout 可以,而且理由比「更快」硬。** rollout 是纯自回归生成 + temp=0,
投机解码在贪心验证下**无损**。这一点是承重的:我们在用草稿加速生成
**训练下一代草稿的数据**,任何有损环节都等于拿自己的错误自我污染。

⚠ 但**数学无损 ≠ 逐 bit 相同**:挂草稿改变每步 batch 形状 → 浮点归约顺序变 →
近似平局处 argmax 可能翻。而本栈**已知过度批处理会吐垃圾**(bf16 两节点 KV 溢出,并发 ≤64)。
故 (c) 必须先 diff。

期望别过高:rollout 跑 conc 64,接近我们实测的 **conc48 1.30~1.38×**,
而草稿占 HBM 会挤掉 KV cache、迫使并发下降,**净值可能只有 ~1.2×**。

**HS 转储不能 —— 脚本自己已经写死:**

```
serve_dsv4_a3_singlenode.sh:138
  [ -n "$DRAFT" ] && { echo "!! HS_DUMP=1 is mutually exclusive with DRAFT ..."; exit 2; }
:135  "NO draft: target-only prefill"
```

HS 转储是**纯 prefill**:训练序列的 token 全部已知(就是 rollout 的输出),
目标模型只前向一遍抓 40/41/42 层的 aux hidden。**投机解码加速的是 decode,
这里没有 decode,没有任何东西可投机**,挂上去只会白占显存。

⟹ **那 35% 的训练瓶颈(29.7% EP all-to-all straggler + 5.1% fetch stall)是 prefill 吞吐问题**,
解法是分析器已列的三条:更多 DP 副本 / `EAGER=0` 上图 / 更大 batch,外加预取。与投机解码无关。

---

## 13. 机制的数学身份:线性链 CRF、CP 分解,以及为什么是二元势(2026-08-21)

§8 讲了三项怎么算,这一节讲**它们是什么**。写给以后要跟人解释、或者要决定"下一步往哪加"的人。

### 13.1 线性链 CRF

独立预测每个位置会产出"每个位置单看都对、连起来荒唐"的序列(经典例子:`New`→LOCATION、
`York`→PERSON)。CRF 的做法是给**相邻标签的搭配**也打分:

```
score(序列) = Σ_t unary_t(y_t)  +  Σ_t ψ(y_{t-1}, y_t)
                一元势/发射            成对势/转移
```

**线性链** = 成对势只耦合相邻位置 ⟹ 精确解码从 `O(K^T)` 降到 `O(T·K²)`(Viterbi)。
**条件** = 势是在输入条件下算的(区别于 HMM 建模 P(x,y))。

| CRF | 我们 |
|---|---|
| 位置 t | 块内第 t 个草稿位(0…4) |
| 标签 y_t | 一个 token |
| 标签数 K | **V = 129,280** |
| 一元势 | `U_t(b)` = 块并行前向的 logits |
| 成对势 | **Markov 头的 bias** |

★ 差别全在 K:教科书 CRF 的转移表是 `[20,20]`;我们是 `[129280,129280]` = 1.67e10,存不下,
所以低秩分解 `M = A·Bᵀ`(秩 256,压缩 250×)—— **这就是 Markov 头的来历**。

⚠ **我们只借了 CRF 的结构,没借它的训练和解码:**

| | 教科书 CRF | 我们 | 为什么 |
|---|---|---|---|
| 训练 | 全局归一化对数似然(对所有路径求和) | 逐位置 CE/TV,局部归一化 | 对 129280⁵ 条路径求和不可行 |
| 解码 | Viterbi(最大化整条路径分) | **贪心** | §5 实测 Viterbi **输 0.175 token** —— 投机赚的是最长正确**前缀** `Σ_t P(0..t 全对)`,不是整块总分 |

⟹ 叫它"CRF 形状的打分"比叫它"CRF"准确。

### 13.2 `⊙` 的身份:三阶张量的 CP 分解,不是随手加的 gate

最一般的诉求是一个三路交互张量 `T[a, b, h]`(前驱、后继、上下文),规模 `V×V×d`,存不下。
CP 分解对三阶张量的标准形式是 `T[i,j,k] ≈ Σ_m U[i,m]·V[j,m]·W[k,m]`,代入即:

```
S(a,b,h) = Σ_{m=1..256} A[a,m] · H(h)[m] · B[b,m]  =  ⟨ A(a) ⊙ H(h), B(b) ⟩
```

★ **逐元素乘之所以出现,是因为 CP 分解里三个因子就是逐元素相乘再求和。**
它是三路交互的**最小参数化**,不是一个 gating 技巧。线性映射在结构上无法表达
"a 的作用随 h 变" —— 必须有乘法项。

**像 SwiGLU 吗?** 代数形状同族(都是两个投影逐元素相乘),但用途不同:

| | SwiGLU | 这里 |
|---|---|---|
| 两操作数来源 | **同一个**输入 x | **两个不同的流**(token 身份 / 隐状态) |
| 目的 | 单流内的非线性与容量 | **跨流的条件化** |
| 非线性 | 一支有 SiLU | 两支都是线性 |

更贴切的对标是 **FiLM**(feature-wise linear modulation,`γ(z)⊙x`)——
即只保留缩放项、去掉偏移项的 FiLM。同族还有 LSTM/GRU 门、超网络。

### 13.3 为什么是二元势,不是三元/多元

按重要性排,**infra 只排第三**:

1. ★ **一元项已经看过全部上文。** `U_t = LMHead(h_t)`,`h_t` 是对**整个前缀**前向得来的,
   长程上下文早在一元项里。一元项唯一表达不了的是"**草稿自己**在 t−1, t−2… 选了什么"
   (那些在 h 算完之后才产生),而其中信息量最大的就是 t−1,再往前边际收益掉得很快。
2. **参数代价**:二阶成对势是 token 的三阶张量,每加一阶多一对码本(+66 M)。
3. **解码代价**:一阶 Viterbi `O(T·k²)`,二阶 `O(T·k³)`;贪心也要多带一维状态。
4. **infra**:服务端每步多读一份码本(66 MB,见 §8.7)。

⟹ 真正的理由是 1,不是做不到。**而且可离线量**:用 §5 的重放台把 t−2 也喂进去,看 accept_len 涨不涨。

### 13.4 GDN 式状态转移:已实现,是 `markov_head_type="rnn"`

```python
state = prev_emb.new_zeros(num_blocks, r)
for k in range(block_size):
    z = torch.cat([state, prev_emb[:, k], hidden_states[:, k]], dim=-1)
    gate_raw, cand_raw, out_raw = self.joint_proj(z).chunk(3, dim=-1)
    gate  = torch.sigmoid(gate_raw)
    state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)   # GRU 式门控状态
    outputs.append(self.markov_w2(torch.tanh(out_raw)))
```

用 `O(r)` 的状态携带块内**全部**历史,阶数无上界、代价常数 —— 与 GDN 同思路。

**为什么一直没建议训它:块太短。**

```
块宽 K = 5:  一元项已覆盖块之前全部上文;成对势覆盖 t−1;
             状态能多带的只剩 t−2/t−3/t−4 —— 最多三个 token
```

GDN 那类状态转移的价值在**几千 token** 的尺度上,5 个 token 摊不开那套机器。

★ **但这个判断依赖块宽。** 路线图上有 `block_size=8`;块宽继续涨则这笔账会翻过来,
届时 rnn 头值得重新评估。

⚠ 另一个现实约束:rnn 头**破坏可重放性**(状态依赖实际走过的路径),
`decoder_ablation_probe.py` 明确拒绝在非 vanilla 头上运行(它假设 `bias(a)[b]` 是前驱的纯函数)。
要评估 rnn 头需另建工具。

### 13.5 补:`k` 的含义、三项并存、共享 B 的取舍

**`O(T·k²)` 里的 `k` 不是词表。** 三个尺寸别混:

| | 含义 | 我们的值 |
|---|---|---|
| `T` | 块宽(位置数) | 5(未来 **16**) |
| `k` | **每位置保留的候选数**(格子宽度) | 探针的 `DECODER_K`,跑过 4,计划 16 |
| `V` | 词表 | 129,280 |

全词表 Viterbi 是 `O(T·V²)`=5×1.67e10,不可能;所以先按一元项取 top-k,在 `T×k` 格子上 DP,
相邻两列 `k×k` 条边 ⟹ `O(T·k²)`。二阶要记 (t−2,t−1) 这一对,状态数 `k`→`k²` ⟹ `O(T·k³)`。
**我们服务端用的是全词表贪心 `O(T·V)`,不建格子 —— k 只出现在离线重放台上。**

### 各项可以并存,而且该并存

打分是加性分解的,各项不互斥:

```
S_t(a,b) = U_t(b) + ⟨A(a),B(b)⟩ + ⟨A'(a)⊙H'(h_t),B'(b)⟩ + ⟨C(state_t),D(b)⟩
             一元      无条件bigram        上下文调制             块内历史
```

★ 保留 vanilla 比让新项取代它更好,理由同 §7.3 的加性设计:

| 项 | 学什么 | 样本效率 |
|---|---|---|
| vanilla bigram | 全语料平均的转移统计 | **高** —— 稠密统计,学得快、稳 |
| select | 三路交互 | 中 |
| rnn / attention | 块内历史 | 低,且只在块够长时才有东西可学 |

让新项取代 vanilla = 逼它顺便重学无条件 bigram,浪费容量、收敛更慢。留着,新项只学残差。

### ⚠ 共享后继码本 B 的取舍(上线时才需要决定)

服务端最贵的是 `[V,r]` 的 V 宽投影(每步 66 MB,见 §8.7)。**共享同一个 B 的项可以先在
r 维相加、只投影一次**(rnn 头现在就共享 `markov_w2`):

```python
分立 B:  bias = B @ A[prev] + B' @ (A'[prev]*H(h)) + B'' @ C(state)   # 每步 3 次 V 宽 GEMV
共享 B:  bias = B @ ( A[prev] + A'[prev]*H(h) + C(state) )            # 每步 1 次
```

| | 分立 B | 共享 B |
|---|---|---|
| V 宽 GEMV / 步 | 每项一次 | **一次** |
| 零初始化 / 独立消融 / 优雅退化 | **能**(SelectHead 靠这个) | 不能,项之间纠缠 |

⟹ **做实验用分立,上线再考虑共享;且不能事后合并**(分立训出的 B 与 B′ 张不出同一空间)。

### 廉价的前置实验:是"信息缺"还是"误差累积"

后段位置弱(p1 0.688 → p5 0.282),两种解释:缺 t−2 及更早的信息,或误差沿前缀累积。
**用 §5 重放台把 t−2 显式喂进成对势,看 p3/p4/p5 涨不涨** —— 涨则信息确实缺(高阶/状态/attention 值得做),
不涨则是累积问题,加什么阶都没用。不占卡、不用训练,且它决定后面所有加项要不要做。

---

## 14. ★ 块宽 16 的语境下,上面的结论要整体改写(2026-08-21)

项目未来要做 **一次投机 16 个**。这个前提推翻 §13.4 "块太短所以 rnn 不划算"的判断,
而且推翻的方式比"历史更长了"更根本。

### 14.1 K=16 时一元项在后段基本失效

`U_t = LMHead(h_t)`,而 `h_t` 来自**任何选择发生之前**的块并行前向。t=15 时它要在完全不知道
token 0…14 的情况下预测第 16 个 —— 一元项在那里几乎没有信息。

```
我们当前 run(K=5)        p1 0.688 → p5 0.282
上游 #12968 官方草稿(8位) 94, 87, 80, 74, 68, 62, 57, 51   ← 第 8 位已掉到 51%
```

⟹ **K=5 时一元项是主力、成对势是修正;K=16 时后段反过来 —— 序列结构才是主力。**
这也意味着 §13.3 "一元项已看过全部上文,所以二元够用"那条论证**只在小块宽下成立**。

### 14.2 小 attention:K=16 下严格优于 RNN/GDN

| | 状态压缩(RNN/GDN) | attention |
|---|---|---|
| 每步代价 | `O(1)` | `O(t)` |
| 表达力 | 全部历史压进 `r` 维**瓶颈** | 内容寻址,**无压缩损失** |
| 适用尺度 | **几千 token** | **几十 token** |

**GDN/线性注意力存在的理由就是 attention 在长上下文上太贵。16 不是长上下文** ——
`K²=256` 个 kv 对,算力可忽略。在这个尺度上没有理由为省算力去背一个压缩瓶颈。

⟹ **K=16 下小 causal attention 比 rnn 头更对。** §13.4 推荐 rnn 是 K=5 前提下的结论,前提已变。

### 14.3 ★★ 这个模块我们已经有了 —— 就是 Correction 头

```
--correction-num-layers 1   "Number of causal Transformer layers in the correction head"
--correction-num-heads  8   "Attention heads in each correction layer"
previous_logits_down/proj   "previous probs[V] -> rank -> correction state -> rank -> bias[V]"
```

**Correction 头就是一层 causal transformer、跨块内位置、消费前一位置的信息** —— 正是"块内小 attention"。

而它在 K=5 上的结果是 **+0.017,收敛到约等于没有**(§1)。

★ **但那个结论不能外推到 K=16。** 5 个位置上一个 causal attention 几乎没东西可看
(一元项已把活干完)。**它的失败是"没机会",不是"没能力"。**

⟹ **`block_size` 提到 8 或 16 的那次重训,是重新评估 Correction 头的自然时点**,
而且代码现成(`feat/dspark-next-port` 已移植并跑通)。届时的候选组合是:

```
U_t  +  vanilla bigram  +  select(上下文调制)  +  小 causal attention(块内历史)
                                                  ↑ 三者并存,不是二选一
```

⚠ 前置:§13.5 那个"t−2 探测"应当先做 —— 它区分"信息缺"与"误差累积",
是上面所有加项值不值得做的共同前提。

---

## 15. 官方 DFlash2 参考实现:块内动态因果卷积(2026-08-21)

### 15.1 先纠正我自己的一处误判

我曾说 vllm-ascend #14533 的 conv 实现"与官方有实质性差异"。**那是我的误读** ——
第一次只抓到 `_grouped_conv` 这个 helper、没抓到调用点,就推断成"骨干输出后加一层"。
拿到调用点后确认:它也是 `attention_conv`/`mlp_conv` 两组 + `prepare`/`finish`,**与官方同构**。

两边数学也等价:

```
官方        output += base[o]*values;  output = addcmul(output, dynamic[o], values)
              ⟹ Σ_o (base[o] + dynamic[o]) * values
vllm-ascend coefficients = base + delta;  output += coefficients[tap]*shifted*(position>=tap)
              ⟹ 同式
```

**唯一差别是布局**:官方 `[B, L, D]` 靠 batch 维天然隔离块;vllm-ascend 拍平成 `[T, D]`,
用 `position % block_size` 掩码补偿(`p < tap` 时屏蔽上一块的尾巴)。**等价,不是错误。**

⟹ 训练侧用官方写法(无掩码);**服务端必须用掩码写法**(那边张量拍平且有 ACLGraph padding)。

### 15.2 官方源:`github.com/z-lab/dflash`

从 HF 模型卡拿到(`z-lab/Qwen3.8-27B-DFlash2`)。模型卡直陈用途:

> *"two-tap dynamic convolutions in the backbone **keep the draft from decaying toward the
> end of the block**."*

**这正是我们的问题**:本轮 p1 0.688 → p5 0.282;#12968 里官方发布草稿到第 8 位已 51%。
而且块宽 16 时更严重(§14.1)。

官方配置(`config.json`):

```
block_size 8 · selector_rank 256 · selector_top_k 16 · conv_kernel_size 2 · conv_group_size 16
无任何 markov 字段 —— 只有一个 selector
```

参考实现 `dflash/model.py`:

```python
def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    blocks = hidden.view(batch, length, groups, group_size)      # [B, L, D] 三维,天然分块
    out = torch.zeros_like(blocks)
    for offset in range(taps):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0,0,0,0,offset,0))
        out = out + base[offset].view(1,1,groups,group_size) * values      # 静态,per-channel
        out = torch.addcmul(out, dynamic[:,:,offset], values)             # 动态,per-group
```

`base_kernel[2, kernel_size, hidden]` 的 **2 = prepare / finish 两侧**;
`kernel_projection` 在 `prepare` 里**一次算出两侧系数**,`finish` 复用 ⟹ 每个子层只投影一次。

### 15.3 位置:与 mHC 的对应(选定 B)

```
官方(普通残差)                          我们(mHC)
residual = h                              residual = streams        [N,γ,hc,D]
h = input_layernorm(h)                    post,comb,x = attn_hc(streams);  x = attn_norm(x)   [N,γ,D]
h,k = conv.prepare(h)                 ★   x,k = attn_conv.prepare(x)
h = sublayer(h)                           x = attn(x, ...)
h = conv.finish(h,k)                  ★   x = attn_conv.finish(x,k)
h = residual + h                          streams = place(x, residual, post, comb)
```

`place()` 的 docstring 明写 `out [B,S,D]` ⟹ `x` 天然是 `[N, γ, dim]`,
**正是官方 conv 要的形状,不需要 reshape、不需要 mask**(块由 batch 维隔离)。

⚠ 一处 mHC 特有的后果:`place` 会先把子层输出乘上 `post` 再折回多流,官方是直接相加。
恒等初始化时无影响,但**该模块的输出在下游被重新缩放** —— 梯度看着偏小时想起这条。

### 15.4 落地(`6ea8899`)

```
新增  backbone/block_conv.py     GroupedDynamicCausalConv,逐比特对齐官方
改    backbone/block.py          位置 B,两处;ks=0 时 conv=None ⟹ 前向逐比特不变
改    config ×2 / weights.py / train.py / launcher
旗标  BLOCK_CONV_TAPS=2  BLOCK_CONV_GROUP=16      默认全关
```

**恒等初始化**:零延迟静态系数 = 1、其余 tap = 0、动态投影置零 ⟹ step 0 与不加 conv 逐比特相同。
五项验证:恒等 ✓ / 梯度到两边 ✓ / 块内因果 ✓ / 块间隔离(无 mask)✓ / **对官方参考逐比特 ✓**。

参数:每组 4.21 M × 每层 2 组 × 3 层 = **25.3 M**(对比 select 一对码本 66 M)。

★ **而且服务端不额外读码本** —— 它在本来就要做的并行前向里,不像 markov/select 每步要付
一次 V 宽 GEMV(conc1 下 66 MB/位置,§8.7)。**同一个弱点,两种机制,价格差一个量级。**

### 15.5 单头收敛:vanilla 与 select 没有理由共存

`H'(h) ≡ c` 时 `⟨A'(a)⊙c, B'(b)⟩` 就是一个固定低秩双线性型 ⟹ **vanilla 是 select 的严格特例**。
官方 config 也印证:**只有 selector,没有 markov**,两者是同一槽位的两代做法,不是叠加。

预算对比(V=129280):

| | 码本 | V宽 GEMV/位置 | 访存 |
|---|---:|---:|---:|
| 现在 vanilla(256)+select(256) | 132.4 M | **2 次** | 264.8 MB |
| 单个 select rank 512 | 132.4 M | **1 次** | 264.8 MB |
| 单个 select rank 256(官方档) | 66.2 M | 1 次 | 132.4 MB |

我原先主张加性双头的三条理由(优雅退化 / 结构性消融 / 服务端可先落地)**都是实验脚手架**,
不是最终设计的属性;第四条"样本效率"最弱 —— `H'` 用 `weight=0,bias=1` 初始化时,
**单头本来就从纯 bigram 起步**,"先学简单的"在一个头里自然发生。

⚠ **但 rank 512 没有直接证据**:官方在 V=248320(近我们两倍)上只用 **256**。
所以收敛到单头是对的,**默认应先试 rank 256**,512 作为 A/B 的另一臂。

⚠ **不要动正在跑的 run** —— 加性双头正是让它 A/B 可读的东西。采纳时点是 `block_size` 8/16 那次重训。

---

## 16. ★ SELECT 判决:空结果。以及为什么下一步是 conv(2026-08-21)

### 16.1 判决

`faithful_ep_20260820_231654`(SELECT_RANK=256,从零训)对配对基线
`faithful_ep_20260804_165215`(ropefix),25844 步时读数:

```
        5000    9000   13000   17000   21000   25000     误差带
SELECT  +0.068  +0.034  +0.028  +0.022  +0.033  +0.020   ±0.079
CORREC  +0.062  +0.011  +0.019  +0.015  +0.033  +0.022   ±0.080
```

**末段 +0.020 ± 0.079。** §9.6 预注册门槛「≤ +0.03 = 死」⟹ **SELECT 死。**

预注册时我的预测是 **+0.053(边缘)**;**实测比预测还差**。记在这里,免得以后把"我早就说
它不行"当成先见之明 —— 当时的判断是「大概率边缘,值得跑一次看」,实际落在更下面一档。

### 16.2 ★ 最硬的证据不是那个数,是两臂逐点重合

SELECT 和 CORRECTION 是**毫不相干的两个机制**(一个是转移项的低秩交叉项,一个是块后
1 层因果 Transformer),却在**六个采样点上逐点重合**,末段差 **+0.000 ± 0.113(0.00σ)**。

而且 SELECT **精确复刻了 Correction 的形状** —— 早期 +0.068、单调衰减、收敛到 +0.02。
Correction 当初正是我们立的校准标尺(§12.1:「早期上涨什么都不能说明」),**SELECT 一步
不差地走完了同一条路**。标尺经受住了它的第二次检验。

⚠ **误差带 ±0.113 偏保守。** 两臂同种子同数据、batch 序列逐步一致 ⟹ 逐 batch 噪声是
**相关**的,配对差的方差远小于两个独立带的合成。也就是说真实可区分度比工具报的**更低**,
结论只会更硬。**工具改进项**:两臂比较应走配对差序列,不是独立带合成。

### 16.3 机制线索:降 loss,不改 argmax

`d_loss` 每一点都是**负的**(SELECT −0.013…−0.060),`accept_len` 却不动。

我们是 greedy / temp=0(见 [[dspark-rollout-greedy-temp0]]),接受与否只看 argmax。
markov-select 和 correction **都是在固定表征上做重排/打分**:把正确 token 的概率抬高了
(所以 loss 降),却没把它从第二名顶到第一名(所以 accept_len 不动)。**这类项对
accept_len 近乎不可见**,而这正是两次实验都空的共同原因。

⟹ **conv 是另一种东西:它改的是 `h_t` 本身**,即生成候选的特征,不是候选的分数。
它至少**有资格**改变 argmax。这是选它作为下一步的机制理由,不是"还没试过所以试试"。

### 16.4 为什么在 K=5 试,而不是等 block-16

我最初主张把 conv 并进 block-16 重训(理由:官方说 conv 治的是"块尾衰减",K=5 下这个
机制最弱)。**用户否掉了,理由成立且我算错了账**:

```
K=5  复用现成的 08-04 配对基线   = 1 次训练,配对关系还在
K=16 基线也得重跑               = 2 次训练,且无配对
```

**成本差一倍,而且 K=16 那次拿不到干净的 A/B。** 先在 K=5 拿一个便宜的读数是对的。

### 16.5 本轮启动命令(CONV 臂)

与 canonical **逐字段相同,只把 `SELECT_RANK=256` 换成两个 conv 旗标**:

```bash
cd /home/a00652497/dspark_austin/speculators
git checkout dflash2-reproduce && git pull

DSPARK_EP=1 BF16_EXPERTS=1 RECOMPUTE=1 COMPILE=0 \
DSPARK_MOE_BALANCE=1 DSPARK_MOE_BALANCE_RATE=1e-3 DSPARK_LOG_EXPERT_LOAD=1 \
INIT_LAYER=1 INIT_MOE_NO_ROUTER=1 \
BLOCK_CONV_TAPS=2 BLOCK_CONV_GROUP=16 \
LR=2e-4 EPOCHS=5 MAX_ANCHORS=512 CKPT_FREQ=0.5 \
DATA=/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow_0730_77w_dedup \
  bash examples/ascend_npu_dflash/train_dsv4_dspark.sh faithful
```

走开前必须确认(与 SELECT 那次同一个坑,`--from-pretrained` 会静默吞掉旗标):

```bash
head -40 $RUN/faithful_ep_<新TS>.log | grep -i block_conv   # 须见 block_conv_kernel_size=2
```

以及前几十步 `hard_accept_len` 与基线同量级 —— conv 是**恒等初始化**,起步就该是恒等;
崩了说明没生效或插错位置。

盯显存:conv 多 25.3 M 参数,但真正吃的是 `kernel_projection` 的激活,
每站 `[B·L, 4096] × [4096, 1024]`,**六站**(3 层 × 2 站)。`RECOMPUTE=1` 在。

### 16.6 ★ 这次的主指标不是总 accept_len,是逐位置准确率

官方对 conv 的原话是 "keep the draft from decaying **toward the end of the block**"。
所以总量之外必须看 `train/position_{k}_acc`:

```
pos4/pos5 抬得比 pos1/pos2 多   ⟹ 机制对上了。即使总量只 +0.02,也说明 K=16 下会放大
全位置均匀微抬                 ⟹ 又一个「降 loss 不改 argmax」,与 §16.3 同类
```

**同时读基线的逐位置**(零成本,`analyze_train_run.py $RUN/faithful_ep_20260804_165215.log`):
知道 K=5 下块尾到底掉多少,才能判断 +0.02 是「机制无效」还是「机制有效但 K=5 空间就这么点」
—— **这两者对 K=16 的推论完全相反**,而这正是这次实验最值钱的产出。

### 16.7 分析器:一个真 bug,但它不是那次挂死的原因

`checkpoint_steps()` 是**二次**的:每遇一个 `Saving checkpoint` 标记就把全文切一刀
(整块内存拷贝)再全文正则扫一遍,收集全部 `global_step=` 只为取最后一个。
代价 = `标记数 × 日志字节数`,两者都随步数线性增长。已改为单次前向扫描(`73ee372`),
6k–26k 步合成日志上逐位等价,258 存档 / 15.8 MB 时快 26×,倍数随日志大小线性长。

⚠ **但它不是 25844 步那次挂死的原因。** traceback 指向 `load()` 的 `_PAIR.findall`,
在 `checkpoint_steps` **之前**;真实原因是冷页缓存下首次读大日志慢,`^C` 按早了。
**工具当时没坏,只是慢。** 记下来,免得以后把这个修复当成"修好了挂死"。

---

## 17. CONV 臂起跑,以及起跑时暴露的两个流程缺陷(2026-08-21)

### 17.1 run

`faithful_ep_20260821_225635`,`--block-conv-kernel-size 2 --block-conv-group-size 16`,
从零训 5 epoch。旗标已双重确认:`pgrep` 的进程命令行、以及 `train_command.txt`
(其 `Git SHA: 6114b56` 与树一致 ⟹ 这个 checkpoint 可精确复现)。

配对基线两条,都要读:

```
CONV vs faithful_ep_20260804_165215   与历史可比(ropefix)
CONV vs faithful_ep_20260820_231654   ★ SELECT 已判空 ⟹ 它就是「什么都没加」的对照线,
                                        且同数据、同 init 旗标,只差一个头
```

### 17.2 ⚠ 缺陷一:探针会覆盖训练 run 的 `train_command.txt`

`scripts/train.py:702` 的 `save_train_command(args.save_path)` 在 rank0 无条件写入。
我们后来用 **`LR=0` 探针**对着同一个 `--save-path` 复跑检查点时,它**又写了一遍** ——
`ckpt_faithful_ep_20260804_165215/train_command.txt` 现在装的是
`decoder_ablation_probe.py --lr 0 --max-anchors 128 ...`,**08-04 那次训练的复现存档已丢失**。

后果不止丢档:我们据它做「只差一个变量」的 diff,得到 8 处差异,其中
`arrow_0720_77w` vs `arrow_0730_77w_dedup` 一度像是数据集混淆 ——
**是假警报**,探针自己的参数。真话在日志里:`grep -o "arrow_[a-z0-9_]*"` 三条 run 全是
`arrow_0730_`,**无数据集混淆**,§16.2 的结论不受影响。

**修法(未做)**:探针类脚本应指向独立 `--save-path`;或 `save_train_command` 遇到
已存在文件时改写 `train_command.<ts>.txt` 而非覆盖。

**教训**:验证「两次 run 只差一个变量」时,`train_command.txt` **不是可信来源** ——
任何人对着同一个 save-path 跑过东西,它就被改写了。可信来源是日志本身。

### 17.3 ⚠ 缺陷二:`hard_accept_len` 早期恒等检查是废的

我给的走开前检查是「前几十步 `hard_accept_len` 与基线同量级」。实际输出是连着三十行
`hard_accept_len=1.000` —— **1.000 是地板**,从零训的开局不管有没有 conv 都是这个数。
**这个门区分不了任何东西。**

有效的替代:头对头的 **soft `accept_len` 差**。@200 步读到
`ROPEFIX 1.037 / CONV 1.031, Δ −0.006` —— 噪声量级,与恒等初始化一致。

⚠ 同时 `loss +0.248`(2.074 vs 1.826)偏大。早期 loss 从 5.12 陡降到 2.05,该区间任何
微小错位都会放大,**200 步不下结论**;2500 步若不收拢,回头查 §15.3 记的那条:
`place` 会用 `post` 缩放子层输出,而官方是不缩放直接加。

### 17.4 起跑时的 serve-bound 读数(待复核)

```
                    基线      CONV
HS fetch %          0.1%     27.4%
HS straggler %     35.7%     25.4%
serve-bound 合计   35.8%     52.8%     effective 4503ms vs steady 2090ms = 2.2x
```

工具判词 `HS/SERVING-bound`,真实 fwd 仅 1%。**但开局现象为主**(启动 217s 计入,
真实 HS 停顿集中在 steps [-1,1,2,14] = 滚动缓冲刚开始填)。
**跑满一个 epoch 后加 `--skip` 重看再定。** 若仍 ~50%,对策是抬 `--max-anchors`
(让每步更重以匹配 serve 产出速率),**不是动 conv**。

---

## 18. 新机器:DSV4-Flash **w8a8** 单机 A2 服务栈,从零到 2.35×(2026-08-22)

与前十七节是**另一条线**:那些是训练侧的草稿改进,这节是服务侧的部署。同一天并行进行,
互不占用机器(109 在跑 CONV,115/116 在供 HS,这是第四台)。

### 18.1 机器与栈

```
登录用户  f00518697       ⚠ 仓库在共享账号下:/home/a00652497/2026/dspark/speculators
conda     /home/f00518697/miniconda3/envs/dsv4-w8a8      ← 两个不同的家目录
权重      /data/ckpt/DeepSeek-V4-Flash-0731-w8a8         293 GB / 74 分片 / 107851 张量
CANN      9.1.0.0627 (beta.3)   ⚠ 9.0.0 不行,见 18.2
vLLM      v0.27.1
vllm-ascend  上游 main @ 4ce367a(#14696 的合入 commit = 地板)
```

**为什么必须是这三个近期合入**(我们原来的 v0.23.0 一个都没有):

```
#12968  08-20  MRV2 支持 DSV4 DSpark          877 → 1192 tok/s(纯引擎侧)
#14490  08-21  修 dsv4 量化名不匹配            量化部署直接踩
#14696  08-21  修 MRV2 上 DSpark 两处崩溃      ★ 当天上午才合,即地板
```

⚠ #14696 合入当天就被我们用上,**MRV2+DSpark 基本零 soak**。出怪事先怀疑它,MRV1 是退路。

### 18.2 ★ 环境六坑,全部已固化进脚本

新机器照 `install_npu_env_dsv4_w8a8.sh` 跑不会再撞。按撞上的顺序:

| # | 症状 | 真因 | 处置 |
|---|---|---|---|
| 1 | `CondaHTTPError: 403` | Anaconda `defaults` 频道许可证门禁;**失败后不留 env**,报错推迟到下一条 `activate` 才以 `EnvironmentNameNotFound` 出现 | `-c conda-forge --override-channels` |
| 2 | `No module named pip` | conda-forge 的 python 不像 defaults 那样带 pip | 创建时显式要 `pip`;脚本内 `ensurepip` 自愈 |
| 3 | `CXXABI_1.3.15 not found` → `Failed to load torch_npu` | conda-forge 的 `libsqlite` 带 ICU 扩展,`import sqlite3` 拉 `libicui18n.so.78`,而链接器命中系统旧 `libstdc++`(CANN 的 set_env 把系统路径排前面) | `$CONDA_PREFIX/lib` 提到 `LD_LIBRARY_PATH` 最前 |
| 4 | 23 个 `错误:'aclmdlRITask' was not declared` | **不是 vllm-ascend 的代码**,是 torch-npu `2.10.0.post4` 自带的 `acl_rt.h` 引用了 CANN 9.1.0 才有的类型 | 升 CANN 9.1.0(唯一要动手装的) |
| 5 | 升级后仍挂在同一处 | `set_env.sh` 是**前置**到 PATH 不是替换,旧 CANN 仍在前面 | 脚本比对 `ASCEND_HOME_PATH` 与目标 CANN,不符**当场退出**(否则白编 40 分钟) |
| 6 | `ZSH_VERSION:未绑定的变量` | CANN 的 `nnal/atb/set_env.sh` 读 `$ZSH_VERSION`,而我们开着 `set -u` | source 时临时 `set +u` |

外加两个非致命但会误导的:
- `libjemalloc.so.2 cannot be preloaded` —— 官方命令写的是 Debian 的 multiarch 路径,
  这台是 RHEL 系(库其实在 `/usr/lib64`)。改成 `ldconfig` 查找后真正用上了。
- `hs_connectors` 缺失让 `import speculators` 挂 —— 它是 uv workspace 成员,`--no-deps` 跳过了。
  而且从仓库根目录跑时,Python 把 `hs_connectors/` **目录**当命名空间包找到,报
  `cannot import name FileTransfer ... (unknown location)`,**读起来像装坏了而不是没装**。
  ⟹ 服务环境根本不需要 speculators(它只用于转换我们自己的草稿),自检改为不因它失败。

**⚠ 两处上游 pin 互相矛盾,只能二选一:**
```
fastapi        vllm-ascend 要 <0.124.0,vLLM 0.27.1 要 >=0.133.0   ⟹ 保 vLLM 的(它在服务 HTTP)
transformers   vllm-ascend 要 ==5.14.1,vLLM 只要 >=5.5.3          ⟹ 取 5.14.1(两边都满足)
scipy/numpy    triton-ascend 钉 scipy==1.13.1,我们钉 numpy 2.3.5  ⟹ 留着警告,服务路径不用 scipy
```

### 18.3 ★ 配方不能跨机型、跨量化照搬 —— 三处代价

**(a) 我最初照的是 A3 的命令。** `DeepSeek-V4-Flash-DSpark.md` 的单机命令是 A3(16 卡,DP4×TP4);
A2 的配方在**另一份文档** `DeepSeek-V4-Flash.md` 的 "A2 series with dspark" 里,我没找到,
自己手推了拓扑。代价不只是并行度错,还漏了两个更要紧的设置。

**(b) `--no-disable-hybrid-kv-cache-manager` 才是长上下文的钥匙。**
DSV4 是混合注意力(Compress-4 / Compress-128),混合管理器按层型分配 KV;不开就按最坏情况
给每层分配:
```
不开:Available KV cache memory 4.84 GiB,需 17.85 GiB,建议 max_len 8794
开  :17.41 GiB,KV 池 1,036,629 tokens
```
**这看起来像并行度问题,其实不是。** TP=8 只多腾约 9 GB,补不上这个数量级的缺口。

**(c) 官方 A2 命令的 `--max-model-len 800000` 是 w4a8 的数。**
它服务的是 `DeepSeek-V4-Flash-DSpark-w4a8-test`,权重约一半。我们 w8a8 + 草稿是 38.99 GB/卡,
同样 800000 在 `gpu_util=0.9` 下**差 0.13 GiB**(引擎自估上限 780288)。
⟹ `GPU_UTIL=0.95` 解决(0.95×60.96 − 38.99 − 1.9 = 17.03 GiB > 14.05)。
**上下文长度不跨量化迁移。**

**拓扑结论(用户先提出,数据支持):**
```
DP4 × TP2   每卡 46.0 GB,而权重共 293 GB ⟹ 驻留 368 GB,约 75 GB 是复制
DP1 × TP8   每卡 35.95 GB = 293/8,零复制;且低并发下延迟更好
```
DP 只在高并发下回本;1–4 人用,三个副本闲着还各占一份非专家权重。
⚠ TP=8 之所以可行,是因为 A2 配方**整块不带 `--additional-config`** ⟹ 没有 `enable_flashcomm1`
⟹ 没有 sequence parallelism ⟹ #14260 那个 "cudagraph 尺寸要同时是 num_spec+1 和 TP 的倍数"
的约束根本不产生。(FC1 就是 vllm-ascend 对非 VL 量化模型的 SP —— 其文档原话是
"an enhanced version of Sequence Parallelism";pass 式 SP 不支持量化。)

### 18.4 三轮基准(conc-1,同一台机同一个栈,逐字段只差投机配置)

```
                tok/s    accept len   acceptance   每步耗时    加速比
AR (NOSPEC=1)    35.1        —            —        28.5 ms      1.00
dspark@5         82.4      4.778        75.6%      58.0 ms    ★ 2.35
dspark@7         77.6      5.326        61.8%      68.6 ms      2.21
```

**num_spec=5 是这份权重在这台机器上的最优点。** 7 的接受长度更高却更慢:
```
accept length  4.778 → 5.326   +11.5%
每步耗时        58.0 → 68.6 ms  +18.3%   ← 涨得更快
第 6、7 位边际接受率 27.4%,而前五位平均 75.6%
```
官方 A2 那行写 7,又是"另一份 w4a8 权重"的数。

⚠ **这三个数不进 eval ledger。** 它们是单条高度结构化的数学题(分数、编号步骤、LaTeX)
重复三次量出来的,而 ledger 里的 3.94 / 4.628 / 上游 4.32 都是 gsm8k 几百条的均值。
**4.778 证明的是"这套栈把草稿跑对了",不是"我们的接受长度更好"。** 要可比的数得跑真评测。

★ 待查:这份草稿记录在案是 `block_size=5`,却能吐 7(1008/144 正好 7)。若原生块宽就是 5,
则第 6、7 位是外推的、对该头分布外 —— 27.4% 的边际接受率正好是这个解释。查 `config.json`
的 `dspark_block_size` 即可判定。**这对我们自己的草稿(训的 block 5、目标 16)意义不同。**

### 18.5 工具(全部在 `examples/ascend_npu_dflash/`)

```
install_npu_env_dsv4_w8a8.sh      ← 18.2 六坑已固化;⚠ 与 install_npu_env_dspark.sh 无关,后者不可动
serve_dsv4_a2_singlenode_w8a8.sh  ← 官方 A2 配方 + TP/MAX_LEN/GPU_UTIL/KV_MEM/NUM_SPEC/NOSPEC 旗标
verify_model_weights.py           ← 起服务前证明权重完整
quick_serve_check.py              ← 一条命令出三个数
serve_traffic_logger.py           ← 记录流量(见 18.6)
```

**`verify_model_weights.py` 当场立功**:74 个分片、293 GB 一个没坏,唯独漏了那个 4 KB 的
`quant_model_weights.safetensors.index.json`(85 个上游文件里就缺这一个)。
少了它 `--quantization ascend` 根本找不到权重。**光看文件数和总大小永远发现不了。**
顺带发现昇腾量化权重的索引名不是 HF 的 `model.safetensors.index.json`。

**`quick_serve_check.py` 的两条设计**(都栽过跟头):
- 先验对错再验速度,而且用"数到 40" —— 重复/跳号/漂移一眼可见,流畅的散文会把错误藏起来。
- **所有请求走对话模板。** 第一版走裸 `/v1/completions`,instruct 模型当文档续写吐了 HTML,
  探针报了假警报;**真正的危害是 accept length 强依赖文本分布**,在离群续写上量的不是我们要的东西。

### 18.6 流量记录:内容留本地,指标必须留服务端

同事都用 harness(Claude Code / Codex / dsh)连这台机器。三个 harness 特有的问题:
```
Claude Code 走 /v1/messages(Anthropic 协议),认证用 x-api-key   ← 只认 Bearer 会直接 401
content 是块结构 [{type:text},{type:tool_use},{type:tool_result}] ← 按字符串抽会抽空
系统提示 + 工具定义每轮全量重发,一次会话几十轮                    ← 逐字记就是 GB/天
```

⟹ 系统提示按摘要**只存一次**(实测 3 KB 系统提示:首条 10501 B,之后每轮 772 B,**13.6×**);
工具结果只留长度 + 300 字头。

**身份分三类存,不合并**:`verified`(API key,对方必须出示,唯一可信)/ `declared`
(`user` 字段、自定义头,随手可改)/ `observed`(IP、UA,NAT 后一堆人共用)。
合成一个字段就把可信的和不可信的搅在一起了。

**★ 最终取舍(用户决定):内容留 harness 本地,服务端只记指标(`--metrics-only`)。**
理由是内容在本地已经是全的,服务端再存是第二份副本;而
```
accept length / ttft / 并发相互影响  ← 只存在于请求发生的那一刻的服务端,事后谁的本地文件都重建不出来
```
`--metrics-only` 一条记录 809 字节,不含任何正文(实测:请求里写"机密内容",日志 grep 不到),
但保留长度、轮数、时延、usage 与投机归因。

⚠ **投机数字只在独占时精确。** vLLM 的计数器是引擎级累计,响应体无 per-request 接受长度;
并发 >1 时差值是几条请求混在一起的 —— 那种记录标 `attributable: false` 并说明原因,
**而不是给一个看起来精确的错数**。

**`conversation_id` 故意设计成纯内容哈希**(系统提示 + 第一条 user),不用随机 UUID:
将来串本地 transcript 时,本地那份同样算一遍就能 join 上服务端指标,两边都不必存内容。

⚠ 前提:大家真的走代理端口。**直连引擎端口的请求没有任何服务端记录,事后补不回来。**
（我们自己做基准要绕开代理 —— 多一跳就多一份噪声。）
