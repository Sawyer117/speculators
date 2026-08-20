# DSV4-DSpark × Correction 头 —— 移植、训练与交付

面向 `dspark_next` 特性集的作者。本文只讲一件事:**我们把 Correction 头那套特性移到
DeepSeek-V4-Flash 的 DSpark 草稿上,跑了多少、看到什么、以及为什么最后的评测得由你来做。**

- 代码分支:`feat/dspark-next-port`
- 基线分支:`feat/dsv4-dspark`(我们已交付的 5-epoch 权重就在这条线上)
- 两者差异(= 移植 + 适配的全部改动):
  `https://github.com/Sawyer117/speculators/compare/feat/dsv4-dspark...feat/dspark-next-port`

---

## 1. 目标模型与基线,先说清楚

| | 我们的 | 你的配置里的 |
|---|---|---|
| 目标模型 | **DeepSeek-V4-Flash**,284B 总参 / 13B 激活 | Qwen3-8B / 0.6B |
| 草稿结构 | 与目标解码层**同构**:MLA(q/o 双 LoRA + per-head sink)、**256 路由专家 + 1 共享**的 MoE、mHC 替代残差 | 稠密 |
| 草稿规模 | 3 层但**总参约 21B、每 token 激活约 1.5B**,训练须专家并行(EP8)+ FSDP2 | 单卡可训 |
| 词表 | 全量 **129,280**,不做草稿词表裁剪 | 32,000 |

基线是我们已完成的 5-epoch 权重 `ep5p0-ropefix`:同一服务栈重测下,五数据集平均接受长度
**4.416**,为官方发布草稿(4.423)的 **99.8%**;非对话四项 4.732,高于官方的 4.692。
**本次 Correction 实验就是与它做配对对照。**

---

## 2. 我们怎么实现的:拿了什么、没拿什么

启动脚本 `examples/ascend_npu_dflash/train_dsv4_dspark_correction.sh` 是一个**薄封装** ——
它只组装旗标,然后原样交给我们已验证的 `train_dsv4_dspark.sh`,因此维度、优化器、调度全部
与产出上述基线的那次运行一致。脚本头 10–52 行是逐条的 PROVENANCE,与下表同源。

### 照搬的(= 被测对象)

`--correction-*`、`--dflash-*`、`--confidence-*` **整块逐字照搬**你的
`dspark_qwen3_8b_sharegpt_online_ascend.sh`。包括 `--correction-output-mode logits`、
hidden-aux-loss 0.1、hidden-feedback、project-corrected-hidden、with-markov,
以及三个 `--dflash-*` backbone 旗标。

### 没照搬的,以及每一条的理由

| 你的 | 我们的 | 为什么 |
|---|---|---|
| `--lr 6e-4` | **2e-4** | 6e-4 在这个栈上 warmup 一结束就 **NaN 发散(约 step 931)**。稠密 4B 扛得住,256 专家 MoE 扛不住 |
| `--loss-fn tv 0.9` | **tv 1.8** | **不是分歧**:speculators 的 `tv_loss` 算的是 TVD = ½·L1,所以 DSpark 论文的 `l1_alpha=0.9` 在这里等价于 **1.8**。用 0.9 相当于半强度 |
| `--block-size 7` | **5** | 为与基线做干净 A/B。本次测的是 Correction 头,不是块宽(块宽是另一条独立的改进线) |
| `--num-layers 5` | **3** | 官方 DSV4 草稿几何 |
| `--target-layer-ids "1 9 17 25 33"` | **"40 41 42"** | 同上 |
| `--draft-vocab-size 32000` | **完全不传** | DSV4 训练在全量 129,280 词表上;传了会建出错词表的头 |
| `--speculator-type dspark` | **dsv4_dspark** | MLA + per-head sink + 256 专家 MoE + mHC |

### ⚠ 一处刻意偏离我们自己的基线

`--no-confidence-detach-features`。我们这个 fork 把 `confidence_detach_features` 默认设为
True,eval ledger 里**每一个检查点都是 detach 训出来的**。你的配置关掉它,本次也就跟着关了 ——
这意味着 confidence 头现在会**反传进草稿**。**若本次结果退步,这是两大嫌疑之一**
(另一个是 Correction 头本身)。这一条我们没有单独做消融。

### ★ 三个 `--dflash-*` 旗标原本在本模型上是死代码

`--dflash-context-residual` / `--dflash-block-position-embedding` /
`--dflash-gated-layer-fusion` 在移植前对 DSV4 草稿**完全没有作用** —— 需要配套把
`dsv4_dspark/core.py::_backbone_forward` 接到继承来的 `_fuse_target_hidden` /
`_condition_noise_embedding` 上,它们才真正生效(提交 `16db370`)。三者均为零初始化,
起步即恒等映射。

### 头宽做了缩放

`--correction-hidden-size` 你用 512(hidden 2560,占 20%);我们 hidden 4096,若照搬 512
只占 12.5%,是**比你调过的更紧的瓶颈**。故按比例放到 **1024**(25%,8 头正好 128/头)。
`--correction-rank` 保持 **256** —— 那是官方发布的 DSV4 草稿在这个 hidden/词表下自己选的
`markov_rank`。代价:头从 78.0M 到 97.1M(+25%),约 +2 个百分点的每 token MAC。

---

## 3. 训练配置与完整启动命令

```bash
DSPARK_EP=1 BF16_EXPERTS=1 RECOMPUTE=1 COMPILE=0 \
DSPARK_MOE_BALANCE=1 DSPARK_MOE_BALANCE_RATE=1e-3 DSPARK_LOG_EXPERT_LOAD=1 \
INIT_LAYER=1 INIT_MOE_NO_ROUTER=1 \
LR=2e-4 EPOCHS=5 MAX_ANCHORS=512 CKPT_FREQ=0.5 \
DATA=<open_perfectblend 自蒸馏 77 万条,0730 去重快照> \
  bash examples/ascend_npu_dflash/train_dsv4_dspark_correction.sh faithful
```

与产出基线的命令**逐字段相同**,只多一个 `DSPARK_LOG_EXPERT_LOAD=1`(纯日志)并换了封装脚本。

数据:提示取自公开的 `mlabonne/open-perfectblend`,应答由 DeepSeek-V4-Flash 以贪心
(temperature=0)重新生成,清洗去重后 775,965 条;**生成与评测全程非思考(non-thinking)模式**。
算力:8×Atlas A2 训练(FSDP2 + EP8)+ 16×Atlas A2 承载目标模型在线供给隐藏状态。

⚠ **初始化**:整层从目标模型第 40–42 层热启动,但 **MoE router 保持随机** —— 继承来的路由
编码的是目标层表示空间中的专家分工,沿用会让负载在训练初期迅速向少数专家集中。

---

## 4. 结果

同种子、同数据 ⟹ batch 序列逐步一致,这是**配对比较**,灵敏度高于跨运行对比。
基线取 `ep5p0-ropefix`(同为损失归一化的 OFF 臂,避免把两个变化搅在一起)。

| 步数 | accept_len Δ(Correction − 基线) | loss Δ | 分析器判定 |
|---:|---:|---:|---|
| 17,077 | **+0.038** | −0.014 | better |
| 24,262 | +0.027 | −0.009 | better |
| 25,879 | +0.023 | −0.007 | better |
| **38,903** | **+0.017** | −0.007 | **~same**(< 0.5% 阈值) |

四点单调,两个指标同向。**我们的读法不是"Correction 头没用",而是"它给了更快的起步,
但两条曲线正在收敛到同一渐近线"** —— 且**这只在训练侧 soft accept_len 这一个口径上成立**。

中止于 step 38,903(1.56 epoch,34.4 小时)。训练全程无 NaN;逐位置准确率全线上行
(p5 0.4885 → 0.5455),无尾部坍塌迹象;MoE 并集三层全饱和(251/255/256)。

### ★ 为什么评测得由你做

```
speculators  src/speculators/models/dsv4_dspark/weights.py   "correction" 出现 0 次
vllm-ascend  vllm_ascend/models/deepseek_v4_dspark.py        "correction" 出现 0 次
```

我们的转换器与推理侧对 Correction 头**零认知**,转出来会**丢掉整个头**;而该模型的推理路径是
`LMHead(h + Δh) + Δlogits + markov_bias`,少了 Δh 与 Δlogits 就是训练/服务不匹配,测出来的
数没有意义。**因此"marginal"这个判断仅基于训练侧指标 —— 真实的服务端接受长度我们给不了。**

---

## 5. 交付物

| 要素 | 位置 |
|---|---|
| 实现代码 | 分支 `feat/dspark-next-port`;与基线的差异见开头的 compare 链接 |
| 启动脚本 + PROVENANCE | `examples/ascend_npu_dflash/train_dsv4_dspark_correction.sh`(头 10–52 行) |
| **训练权重** | `ckpt_faithful_ep_20260818_122129/0` = **1.0 epoch**,`/1` = **1.5 epoch**。标准 HF 格式:`config.json` + `model.safetensors`(EP 的专家是 `Shard(0)` DTensor,上游的 `DistributedCheckpointer` 存盘时已合并为完整 state dict) |
| 本次日志 / 基线日志 | `faithful_ep_20260818_122129.log` / `faithful_ep_20260804_165215.log` |
| 对照命令 | `python3 examples/ascend_npu_dflash/analyze_train_run.py <本次log> --baseline <基线log> --label CORRECTION --baseline-label ROPEFIX --out <目录>` |
| 环境 | `examples/ascend_npu_dflash/install_npu_env_dspark.sh`(唯一权威安装脚本:torch/torch-npu 2.10.0、vLLM 0.23.0、vllm-ascend 从源码编) |

⚠⚠ **跨 epoch 不可直接比。** 我们这两份权重只训到 **1.0 / 1.5 epoch**,而上文那个 4.416 的
基线是 **5.0 epoch** 的。要比就与基线的**同 epoch** 检查点比 —— 全量集五数据集平均分别是
**4.0788(1.0ep)** 与 **4.1800(1.5ep)**。

### 我们在你的分支之上做的适配(共 6 条,**均未改动你的算法**)

```
ef8cf7d  detach 覆盖挪到真正的 config 类;EXTRA_ARGS 透传
16db370  让三个 --dflash-* backbone 旗标在本模型上真正生效(此前是死代码)
92311d2  Correction 头宽按我们的 hidden 缩放(512@2560 -> 1024@4096),rank 保持 256
f48572b  安装并解析 hs_connectors(上游同步后成为强依赖)
d5a9208  让 dsv4_dspark 通过 DSpark 的特性门禁
bcdc2d7  1-D 门控参数 —— FSDP2 拒绝标量
7c25438  昇腾上不走融合 Triton 损失(torch_npu 的 transfer_to_npu 把 .is_cuda 打成 True,
         使 `logits.is_cuda` 这个判据在昇腾上恒真,会去调没有昇腾后端的 Triton 核)
```

最后三条与 Correction 头无关,是上游同步后在昇腾上跑通所必需的,一并列出以便你复核。
