# Qwen3-4B DFlash 训练数据复现

DFlash 草稿模型必须用 **on-policy** 数据训练:开源数据集只提供 prompt,response 要用目标模型
Qwen3-4B 自己重新生成一遍。所以**不能**跳过 rollout 直接拿原数据集分词。

需要的两样东西都是公开的:

| | 出处 |
|---|---|
| 目标模型 | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) |
| 种子数据集 | [`mlabonne/open-perfectblend`](https://huggingface.co/datasets/mlabonne/open-perfectblend) |

硬件:8 卡 Ascend NPU,装好 vLLM-Ascend 的环境。

> ⚠️ **脚本里所有路径默认值都是我们内部共享盘**(`/share/canada_group_folder/...`)。
> 对外复现必须自己传 `MODEL` / `DATASET_PATH` / `OUTFILE` / `TARGET_MODEL` / `TRAIN_DATA` / `DATA_DIR`。

---

## ① 起目标模型的服务(终端 A,保持运行)

```bash
MODEL=<Qwen3-4B 权重目录> \
  bash examples/ascend_npu_dflash/rollout_serve_qwen3_4b.sh
```

默认 8000 端口、8 卡、DP8/TP1、`MAX_MODEL_LEN=3072`、`GPU_MEM_UTIL=0.9`。
不带草稿的纯解码 —— rollout 是批量吞吐任务,纯解码更快。

`DRAFT=<ckpt>` 可以开投机解码,**生成内容完全一致**(投机解码无损),只影响速度;
但它强制 `--max-num-seqs 32` 压低并发,批量场景通常不划算。

[`rollout_serve_qwen3_4b.sh`](../../examples/ascend_npu_dflash/rollout_serve_qwen3_4b.sh)

## ② 生成 response(终端 B,几小时)

```bash
# 先小样冒烟,一分钟内能暴露端口错 / 数据集拉不下来
LIMIT=200 OUTFILE=./rollout_smoke.jsonl \
  bash examples/ascend_npu_dflash/rollout_qwen3_4b.sh

# 正式跑
nohup env OUTFILE=./open-perfectblend.qwen3-4b-rollout.jsonl \
  bash examples/ascend_npu_dflash/rollout_qwen3_4b.sh > ./rollout.log 2>&1 &
```

对齐 SpecForge 的参数已写死在脚本里,不用自己传:贪心 `--temperature 0`、`--no-thinking`
(Qwen3 关思考,否则超长)、`--max-tokens 3072`(= 训练 seq-len,更长也会在 ③ 截掉)、
并发 256(**若 ① 开了草稿要降到 32**)。

想直接从 HF 拉数据,把脚本里 `--dataset-path` 那行去掉,让它按 `--dataset open-perfectblend` 自己下载。

**中断续跑**:底层 [`script.py`](../../scripts/response_regeneration/script.py) 支持 `--resume`
(读输出里已有的 id 跳过),但 wrapper 没暴露,得直接调那个 py。
⚠️ resume 按输入顺序对齐,**改了 `--limit` 或数据集 id 就错位**,其余参数必须一字不动,否则重跑。

**验收**:

```bash
python examples/ascend_npu_dflash/rollout_stats.py <你的 jsonl>
```

重点看 `finish_reason` 里 `length` 的占比 —— 那是撞到 3072 上限被截断的。
几个百分点正常,占比高说明真答案被切了。

[`rollout_qwen3_4b.sh`](../../examples/ascend_npu_dflash/rollout_qwen3_4b.sh) ·
[`rollout_stats.py`](../../examples/ascend_npu_dflash/rollout_stats.py)

## ③ 分词成 Arrow(纯 CPU,不占卡)

```bash
TARGET_MODEL=<Qwen3-4B 权重目录> \
TRAIN_DATA=<② 产出的 jsonl> \
DATA_DIR=<输出目录> \
NUM_WORKERS=32 \
  bash examples/ascend_npu_dflash/prepare_qwen3_8b.sh
```

⚠️ **没有 `prepare_qwen3_4b.sh`,这是故意的。**分词只跟 tokenizer 有关,整个 Qwen3 家族共用
一个 tokenizer,4B 的 Arrow 和 8B 完全相同。别去找 4B 版本,它从没写过。

`NUM_WORKERS` 默认才 8,一定要调大 —— 这步全部时间都在 CPU 上。
取随机子集用 `MAX_SAMPLES=<n> SEED=<n>`,建议把选择写进目录名(我们的 `half50` 就是这个意思)。

[`prepare_qwen3_8b.sh`](../../examples/ascend_npu_dflash/prepare_qwen3_8b.sh) ·
[`prepare_data.py`](../../scripts/prepare_data.py)

## ④ 训练(数据落到哪儿)

产出的 `DATA_DIR` 填进 [`config_qwen3_4b.sh`](../../examples/ascend_npu_dflash/config_qwen3_4b.sh),
然后 `serve_qwen3_4b.sh`(卡 0)+ `train_qwen3_4b.sh`(卡 1–7,7 路 FSDP)。

端口注意:rollout 那对用 **8000**,`config_qwen3_4b.sh` 里训练侧的服务是 **8001**,两者可共存。

---

## 坑

- `rollout_shard.sh` 长得像多机版,但**是 DSV4-Flash w8a8 284B 专用**,参数按那个模型量的,别拿它跑 Qwen3。
- 这些 `.sh` 要 `bash` 跑不能 `source`(脚本会拒);配置文件相反,是给 `source` 的。
- `OUTFILE` 是覆盖不是追加,除非在 resume。
- ② 之前服务必须先起来,脚本不重试连接,第一个请求就失败。
- **`--dataset` 的合法取值是 `open-perfectblend`(连字符)**。注册表 key 在
  [`configs.py:149`](../../src/speculators/data_generation/configs.py) 定义;曾经的下划线写法
  `open_perfectblend` 会被 argparse 直接拒(`invalid choice`),三个 rollout 脚本已在
  `2679037` 修正。旧分支上取回脚本的话先确认这一处。

## 未验证的部分

参数默认值都是从脚本里逐条读出来的。但**这套流程没有在对外环境端到端跑过** ——
唯一实测过的是上面那条 `--dataset` 会被 argparse 拒绝。
第一个真正复现的人可能还会碰到别的内部假设,碰到了请回来补这份文档。
