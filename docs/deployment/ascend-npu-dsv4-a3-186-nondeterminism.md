# A3 186:温度 0 下服务自己和自己对不上 —— 调查归档(2026-09-19 → 09-23)

> **一句话**:这台机在上下文变长时,**同一个 prompt、温度 0、并发 1、prefix cache 关着,
> 连打三次会得到三个不同的输出**。不是崩溃,是**静默算错**。在定性之前,这台机产出的 HS
> 一概不可信,批量 HS 生产必须停。

本文归档整条调查链:每一步在问什么、量到了什么、排除了什么,以及**我在中途说过头、
后来被自己的数据推翻的那些结论**。后者和结论同样重要 —— 它们是这条线上最容易重犯的错。

机器:A3 单机 16 逻辑卡,DP2×TP8/EP16,DeepSeek-V4-Flash bf16。

---

## 0. 起因

把 HS dump 出来的 hidden 过 `lm_head` 取 argmax,和 Arrow 语料里的 token 比,
在 **response 段(`loss_mask==1`)mismatch 高达 64%**。

最初的假设:dumper 写错了层,或者对齐差一位。

---

## 1. 先洗清 dumper:自洽 oracle

不和历史语料比 —— 历史语料是另一套栈、另一个时间产的,拿它当基准等于一次测两件事。
改成只用**服务当场产出的东西**做自洽检验(`examples/ascend_npu_dflash/hs_self_oracle.py`):

| 段 | 检验 | 结果 |
|---|---|---|
| A 恒等式 | 温度 0 时 `serve 吐的 token == argmax(lm_head(它那步的 final hidden))` —— serve 本来就是这么算的 | **4/4 完全一致**,256~2048 每个长度都过 |
| B 回灌 | 让 serve 贪心生成 16 个 token,把 `prompt+生成` 整条回灌做 prefill dump,检查那 16 个位置 | **64 个位置 mismatch 0.00%** |

顺带用 `hs_value_probe.py` 穷举 slice × {raw,normed} × shift{−1,0,1},定死了读法约定:

* dump 形状 `[seq, 3 aux + 1, H]`;
* **`[:, -1]` 是 final 且已经 post-norm** —— 它的 RMS 0.3446 ≈ `norm.weight` 的 RMS 0.3090,
  再过一次 norm 反而更差;
* `argmax(h_i)` 对齐 `token_ids[i+1]`(shift=1 远好于 0/−1);
* `token_ids` 与 Arrow 的 `input_ids` 逐个一致。

⟹ **dumper 的管路是对的。**

> ⚠ 当时我写成了「证明 dump 的**值**是对的」。**这句过头了** —— 恒等式两边都来自同一次前向,
> 两边可以一起错。它证明的是**管路**(层号、切片、对齐、落盘),不是数值绝对正确。

---

## 2. 那就怀疑语料

思路很干净:把 prompt 喂回去让模型贪心续写,和语料里的 response 直接比
(`corpus_provenance_check.py`)。不涉及 HS、不涉及 lm_head、不依赖任何中间约定 ——
只问「模型会不会这么说」。

**然后就撞上了这件事:同一条命令跑两遍,第 0 行从 32/32 变成 0/32。**

温度 0、并发 1、`--no-enable-prefix-caching`。贪心解码本该是确定性的。

---

## 3. 先量服务自己稳不稳

给脚本加了两个开关:`--repeat`(同一个 prompt 连打 N 次互比)和 `--prompt-len`
(忽略 `loss_mask`、直接拿 `ids[:L]` 当 prompt,才测得到长上下文)。

同时**换掉指标**。贪心是混沌的:第一个 token 一分叉,后面全不同,所以
「逐 token 一致率」会把「早分叉一次」和「处处不同」混为一谈 ——
该看的是**完全一致前缀**(第一次分叉在哪)。

同一 prompt 连打 3 次,互比最短完全一致前缀 / 32:

| prompt-len | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| 新栈 `4ce367a` + vLLM 0.27.1 | **32/32** | 2/32 | **0/32** | **0/32** |
| 老栈 `386530d` + vLLM 0.23.0 | 19/32 | 12/32 | 8/32 | 6/32 |

**≥512 就静默算错。** 新栈是断崖,老栈是平滑退化 —— 但**两条都不确定**。

阈值和 **`index_topk = 512` 重合**:≤512 时稀疏选择是平凡的(全选),>512 才真正开始选。
而间歇崩溃的算子正是 `SparseAttnSharedkv`。**崩和静默算错很可能是同一个根因的两面。**

复现:

```bash
ENDPOINT=http://localhost:7000/v1 ARROW=<arrow_0730_77w_dedup> \
python examples/ascend_npu_dflash/corpus_provenance_check.py \
  --n 3 --gen 32 --repeat 3 --prompt-len 2048 --id-base 980000
```

> `--id-base` 必须 > 数据集行数(772,684),否则 HS 会写进生产目录。脚本有硬 guard。

---

## 4. 并行的一条线:`SparseAttnSharedkv` 间歇崩

`dsa_prefill_fault_ab.sh` 四臂 A/B。两个关键结果:

* **arm C(`HS_DUMP=0`)照样崩**,16 次 `SparseAttnSharedkv` ⟹ **和我们的 dumper 无关**。
* **它是间歇的**。同一个臂、同一套配置、都禁了编译缓存:

  | | 用时 | errors | 算子故障 | 引擎 |
  |---|---|---|---|---|
  | 第一次 | 21.9 s | 0 | 0 | 活 |
  | 第二次 | 83.9 s | 231 | 12×SparseAttnSharedkv | 死 |

  ⟹ **判读规则:崩 = 阳性,可信;不崩 = 阴性,不可信。** 阴性只在重复多次全不崩时
  才算弱证据。脚本因此有 `REPEAT`(默认 3)和「臂 X:崩 k/n 次」汇总。

另外查清一件独立的事(`aux` 与编译缓存):`set_aux_hidden_state_layers()` 在 `get_model()`
**之后**才调(`model_runner_v1.py:3601` → `:3661`),所以 aux 配置**进不了 torch.compile
的缓存 key**;而 aux 改的是 forward 的**返回签名**(开了返回 `(hs, aux)`,没开返回裸 tensor)。
⟹ eval serve(无 aux)和 HS-dump serve(有 aux)在同一台机上轮流跑,谁先编译谁占住缓存,
另一个直接命中错误的图。崩掉算走运(`ValueError: too many values to unpack`);
反过来命中「带 aux 但层号不同」的图会**静默 dump 错层**,几 TB 条件输入全错且零报错。
已修:`HS_DUMP=1` 时强制 `VLLM_DISABLE_COMPILE_CACHE=1`。

---

## 5. 站得住的结论

1. **HS dumper 的管路是对的**,别再怀疑它。
2. **算子故障与 dumper 无关**,且是间歇的(~18%)。
3. **服务在上下文 ≥512、温度 0 下不可复现** —— 静默算错,不只是崩。
4. 那个 64% **不是「语料有问题」的证据** —— 量到的就是这个不确定性。
5. **在定性之前,这台机产的 HS 一概不可信,批量 HS 生产必须停。**

---

## 6. 中途说过头、被自己的数据推翻的

记在这里是因为这几条都不是笔误,是**方法上的同一类错**:拿单次结果下结论、
拿另一套栈的数当基准、把「机制对」说成「数值对」。

| 说过的 | 被什么推翻 |
|---|---|
| 「根因是编译缓存污染」 | 基于 n=1 的一次「不崩」。下一次同配置 231 错 |
| 「语料不是本模型贪心输出」 | 我自己后来量到 99.6% 一致 |
| 「并发是变量」 | conc1(88.97%)比 conc64(84.75%)**还差** |
| 「self-oracle 证明 dump 的值是对的」 | 应为「证明**管路**是对的」 |
| 「新栈老栈都不确定 ⟹ 不是栈的问题」 | 见下节 —— 两条臂共用 CANN 9.2.0-beta1 |

---

## 7. ★ 没有排除掉的变量(2026-09-22 修正)

第 3 节那张表我一度读成「**新栈老栈都不确定 ⟹ 不是栈的问题**」。
**这句超出了实验能支持的范围。**

那次老栈是按「老栈就用 CANN 9.2 + 老 VLLM」装的(`install_oldstack_a3.sh` 的注释写着
「只留 vllm-ascend + vLLM 一个变量」——当时是对的做法)。于是:

* 被排除的是 **pin**(`386530d12` vs `4ce367a`)和 **vLLM 版本**(0.23.0 vs 0.27.1);
* **CANN 9.2.0-beta1 一次都没被排除** —— 它是唯一贯穿两条臂的软件变量;
* 而产出 77w 那批 HS 的**真老栈是 CANN 9.0.0**
  (`ascend-npu-dsv4-rollout-data.md:23-24`),我们从没在上面测过。

**还有一条推论值得单独记**:`sas_metadata_buffer`(这个 pin 的 `dsa_v1.py` 把 SAS 每核
任务分配写进一个常驻共享 buffer,`__init__` 分配一次、每步覆写)一度是头号嫌疑。
但旧 pin `386530d12` **没有这个共享**(它把刚算出的张量直接传下去),**却同样不确定**
—— 所以它**解释不了老栈那一半**。这把嫌疑从「新 pin 特有的东西」推向「两条臂共有的东西」:

| 嫌疑 | 状态 | 怎么证伪 | 成本 |
|---|---|---|---|
| **别的机器也这样?**(硬件 / 驱动 / 这台 186) | **没测过** | 换一台机跑同一条 `--repeat 3 --prompt-len 2048` | **几分钟,不用重编** |
| **CANN 9.2.0-beta1** | 没测过 | 装 CANN 9.0.0(或 9.1.0)+ 同 pin | 几小时,要重编算子 |
| `--async-scheduling` | 没测过 | `ASYNC_SCHED=0`,一次重启 | 一次重启 |
| DP2 跨 replica 的 MoE all-to-all(并发 1 时另一个 replica 在跑 dummy batch,形状每步不同) | 没测过 | `DP=1 TP=16` | 一次重启 |

后三项 `determinism_sweep.sh` 三臂(`base` / `noasync` / `dp1`)已经写好,**但一次都没跑过**。

**下一步应当先做最便宜的那一条:换一台机跑同一个测试。** 它一刀把「硬件/这台机」和
「软件栈」切开,几分钟出结果,而重装 CANN 要几小时。

```bash
# 在另一台机上(A2 115/116,或第二台 A3,或 w8a8 那台)
ENDPOINT=http://<那台>:7000/v1 ARROW=<同一份 arrow> \
python examples/ascend_npu_dflash/corpus_provenance_check.py \
  --n 3 --gen 32 --repeat 3 --prompt-len 2048 --id-base 980000
```

读法:

* 另一台机 **32/32** ⟹ 问题在 **186 这台**(硬件/驱动),换机器即可,不用动栈;
* 另一台机**也不确定**且 CANN 也是 9.2 ⟹ 嫌疑锁定 **CANN 9.2.0-beta1**,值得重装 9.0.0/9.1.0;
* 另一台机 CANN 不同却也不确定 ⟹ 是更普遍的东西(`--async-scheduling` / DP / 算子本身),
  回到 `determinism_sweep.sh` 三臂。

---

## 7b. 2026-09-22 03:02–03:21 实跑 `determinism_sweep.sh` —— 头号嫌疑被证伪

机器空闲(16 卡全 `No running processes found`,HBM 空闲基线 2870–3105 MiB)。
三个臂,每点 3 行 × 3 次重复 × 32 token,指标是**完全一致前缀**:

| prompt-len | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| 09-20 新栈(历史) | 32/32 | 2/32 | 0/32 | 0/32 |
| **base**(对照) | **19/32** | 2/32 | 3/32 | 0/32 |
| **noasync**(`ASYNC_SCHED=0`) | 32/32 | **2/32** | **1/32** | **0/32** |
| **dp1**(`DP=1 TP=16`) | — | — | — | **起不来** |

### ① `--async-scheduling` 证伪

512 / 1024 / 2048 三个长度纹丝不动。256 的 32/32 落在噪声里(见下)。
⟹ **`--async-scheduling` × 常驻 `sas_metadata_buffer` 这条假设不成立。**
它此前已经被削弱过一次(旧 pin `386530d12` 没有那个共享 buffer 却同样不确定),
现在是直接实验否掉。serve 脚本的 `ASYNC_SCHED=0` 开关保留,但它不是解药。

### ② ★「阈值 ≈ `index_topk` 512」这个解释作废

同一套栈、同一台机,256 在两天之间是 **32/32 ↔ 19/32**。
所以 09-20 那个「256 完全可复现」是**运气,不是性质**,据它推出的
「≤512 时稀疏选择平凡所以安全」随之作废。
现在的事实是:**所有长度都不确定,只是越长越糟。**
连带:`SparseAttnSharedkv` 仍然可疑,但少了「阈值重合」这条支撑。

**方法上的推论**:单次读数有噪声,判读标准必须事先定死 ——
某臂四个长度**全 ≥30/32** 才算真信号;「看起来好一点」(2048 从 0 变 5)一律算噪声。

### ③ `dp1` 臂是个不成立的实验(我设计错了)

```
RuntimeError: shape '[0, 1024, -1]' is invalid for input of size 2097152
```

前导维是 **0** —— 某个维度被 `TP=16` 整除后向下取整成了 0。bf16 在 A3 上的配方一直是
**DP2×TP8**(官方那个 DP4×TP4 是给 w8a8 的),`TP=16` 从没在这个模型上跑过。
脚本注释里那句「EP 仍是 16,显存只会更宽松:dense 切 16 份」是**未经验证的推断**。

⟹ 这不是根因的证据,是**臂本身不成立**。而且 DP 假设**没法靠去掉 DP 来测**:
`TP=16` 不整除,`DP=1 TP=8` 则 EP 从 16 降到 8、每卡专家显存翻倍(~38 GB → ~76 GB)装不下。
要测 DP,只能换个问法 —— 见下。

### 排除清单(截至 2026-09-22)

| 变量 | 状态 |
|---|---|
| vllm-ascend pin / vLLM 版本 | 已排除(09-20 两条臂) |
| prefix caching | 已排除(`--no-enable-prefix-caching` 一直开着) |
| `--async-scheduling` | **已证伪(09-22)** |
| DP2 跨 replica 的 MoE all-to-all | **仍未测**,且不能靠改并行布局来测 |
| CANN 9.2.0-beta1 | 仍未测(`910env_npu.sh` 在机器上,不用下包) |
| 这台机的硬件 / 驱动 | **仍未测 —— 最便宜,几分钟** |

### 下一步的两个可做实验

1. **换一台机跑同一个探针**(几分钟,不用装任何东西)。一刀切开「这台机」与「软件栈」。
2. **换个问法测 DP**:不改并行布局,而是问「一个温度 0 的请求,它的输出会不会随
   **另一个 replica 在干什么**而变」—— 单发 vs 带背景负载各打三次。会变就直接坐实
   跨请求/跨 DP 污染,而且那本身就是个正确性 bug,与机制无关。

---

## 8. 与「跨栈不可比」不是一回事

容易混,分清:

| | 现象 | 性质 |
|---|---|---|
| **跨栈系统性偏移** | 同一份 released draft,老栈 gsm8k 4.665 / 主线 4.523(−3%),五个数据集五个位置上**都是平移** | 已知、可量、**可比较**(每行自带同栈横杆) |
| **本文这件事** | **同一套栈、同一个 prompt、同一次服务**,连打三次三个结果 | 自身不自洽,**任何单次读数都失去意义** |

所以「切回一致的栈就没事了」这个说法要拆开看:**如果**根因是 CANN 9.2.0-beta1,
那么回到 9.0.0 确实会没事 —— 那套栈跑出过 gsm8k 96.59%。
**⚠ 但那个成绩不是在 186 上量的。** 2026-09-22 复核:没有任何记录把 96.59% 归到 186,
它指向 176(ledger 里 33 处写 `176 A3-single`)或 rollout 机;186 是第二台 A3 评测机,
它自己被记录下来的成绩是 0.28.0+main 的 **92.04%(已否)**。
⟹ **在 186 上装 CANN 9.0.0 不是「恢复一个已验证的好配置」** —— 186 从来没有过一份
已验证干净的配置,那会是个全新的、没人跑过的组合。这也把「换一台机跑探针」的优先级
再往上提了一格:**如果 176 还活着且老栈还在,它就是那个已知良好的对照。**
但这是**一个还没验证的假设**,不是已知结论;而且它解释不了「为什么老栈在 CANN 9.2 上
在 256 长度反而比新栈更差(19/32 vs 32/32)」。先做第 7 节那个几分钟的换机测试。

---

## 附:相关文件

| 文件 | 作用 |
|---|---|
| `examples/ascend_npu_dflash/hs_self_oracle.py` | 自洽 oracle(A 段恒等式 / B 段回灌) |
| `examples/ascend_npu_dflash/hs_value_probe.py` | 穷举 slice×norm×shift,定读法约定 |
| `examples/ascend_npu_dflash/corpus_provenance_check.py` | 语料溯源 + **确定性自检**(`--repeat` / `--prompt-len`) |
| `examples/ascend_npu_dflash/dsa_prefill_fault_ab.sh` | 算子故障四臂 A/B,带 `REPEAT` |
| `examples/ascend_npu_dflash/determinism_sweep.sh` | 三臂确定性扫描 —— **写好了,没跑过** |
| `examples/ascend_npu_dflash/npu_cleanup_lib.sh` | 已核实清场(进程 + 显存双门) |
| `examples/ascend_npu_dflash/install_oldstack_a3.sh` | 老栈安装;`CANN_ENV` / `ENV_NAME` / `ROOT` 可覆盖 |


---

## 9. 2026-09-23 实跑 —— 换 CANN、换 MoE 实现,以及换一个口径

> 本节自包含:数字和口径都在这里。`dsv4-dspark-article` 的
> `experiments/reports/serve-nondeterminism-2026-09-23.md` 有一份同内容的记录,
> 外加机器可读的 CSV —— 是刻意冗余,不是接续。

### 9.0 先说结论

**这台机上没有任何配置提供逐 token 可复现。**但当天最硬的发现不是"有多不稳",而是
**不稳跟「这次 forward 里有多少 token」走,跟位置无关** —— 因果模型里前缀的输出不可能
依赖后面的 token,而实测它依赖(§9.4)。同时**聚合读数仍然稳到 0.014**,历史 eval 数字
依然可引用。

### 9.1 三次崩溃的真因:我们自己配方里的一行死配置

试图关 MegaMoe(`VLLM_ASCEND_ENABLE_FUSED_MC2=0`)时,服务连着三次在 `profile_run` 挂掉:
CANN 9.1 上 `vector core timeout`(507034),CANN 9.2 上 `fftsplus aivector error`。

`serve_dsv4_a3_singlenode_specmethod.sh` 里写死了:

```
ACFG='{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true, ...}'
```

而 vllm-ascend `ascend_config.py:174`:

```python
if self.enable_fused_mc2 == 1 and self.multistream_overlap_shared_expert:
    self.multistream_overlap_shared_expert = False      # 只有 ==1 时才强制关
```

两者互斥、fused_mc2 赢,而我们默认 `FUSED_MC2=1` ⟹ **这行从来没生效过,是死配置**。
一关 FUSED_MC2,这条从未运行过的 multistream 路径第一次被唤醒,于是崩。

⟹ **`FUSED_MC2=0` 不是关一个开关,是关一个、同时开一个。**已修:脚本里它现在跟随
FUSED_MC2(已验证配方里它实际是 OFF,关 MegaMoe 时保持 OFF 才是单变量),`MSOSE=true` 可强开。

**顺带**:`_select_a3_moe_comm_method` 里 `enable_fused_mc2 == 1` 分支在任何 `num_tokens`
判断**之前**就 return;关掉后每 rank 容量上限从 4096 掉到 512,8192 的 profile run 因此落到
**AllToAll**。所以那三次崩溃走的既不是 MegaMoe 也不是 MC2 —— 我当时据此说的
「MC2 在这台机上两个 CANN 都坏」**作废**。

### 9.2 生成层面的四臂对照

口径同 §3:同一 prompt 连打 3 次,量**完全一致前缀**,生成 32 个 token。

| 臂 | 日期 | CANN | FUSED_MC2 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| new-megamoe | 09-20 | 9.2.0-beta1 | 1 | 32 | 2 | 0 | 0 |
| new-megamoe | 09-22 | 9.2.0-beta1 | 1 | **19** | 2 | 3 | 0 |
| old `386530d12` + vLLM 0.23.0 | 09-20 | 9.2.0-beta1 | n/a | 19 | 12 | 8 | 6 |
| new-alltoall | 09-23 | 9.2.0-beta1 | 0 | 19 | 12 | 8 | 20 |
| cann91-alltoall | 09-23 | **9.1.0-beta.3** | 0 | 14 | 32 | 14 | 26 |

- **老栈和新栈跑 AllToAll 时前三格完全相同(19/12/8)。**老 pin 早于 MegaMoe,走的本来就是
  AllToAll ⟹ **同一条代码路径**。§5 里"两套栈都不确定"因此不是两件事,是一件:
  **pin 不是变量,MoE 路径才是。**
- **MegaMoe 是显著加重因素,不是根因**(512:12→2,1024:8→3);AllToAll 自己在 256 上
  也只有 19/32。
- **CANN 9.1 有影响但不单向**:512 打满、1024 变好、**256 反而变差**。当前样本量不能定性。

⚠ **这张表是 min 统计**(9 次两两比较取最差)。单次复现率 80% 的配置,9 次全过只有
`0.8⁹ = 13%` 的概率。实测同一配置两次测量 2048 那格是 **6 和 20**,±14。**只能比大小。**

### 9.3 换口径:纯 prefill

**训推两端共用的不是 decode。**训练侧 HS 是整条语料一次 prefill 抓的;部署侧投机验证是
`[上次接受的 token + γ 个草稿]` 一次 forward,**prefill 形状** —— 主模型在投机模式下根本
不做单 token decode。decode 只出现在语料生成那一次,而那份语料现在是固定输入。

`prefill_noise_probe.py`:同一串 token,`max_tokens=1` 只做 prefill,重复 3 次,逐位置比
`prompt_logprobs`。**位置之间互不污染**,没有 decode 的反馈放大;报的是带 Wilson 区间的
比例,不是 min。

结果(mainline 栈 + MegaMoe,CANN 9.2.0-beta1,8 行 × 前 2048 token × 3 次):

| bg | 行长 | 翻转率 |
|---|---|---|
| 0 | 262 / 276 / 330 | **2.30 / 2.18 / 1.82%** |
| 0 | 587 | 9.73% |
| 0 | 2010 / 2028 / 2048 / 2048 | **94.6 / 99.7 / 91.6 / 82.2%** |
| 4 | 262 / 276 / 330 | 3.45 / 1.82 / 1.22% |
| 4 | 2010 / 2028 / 2048 / 2048 | 44.5 / 74.0 / 31.1 / 19.3% |

按 margin(top1−top2,nat):bg=0 下 `≤0.01` 翻 90.9%、`0.5–2` 翻 83.9%、
**`>2` 笃定档翻 61.8%**;bg=4 下笃定档 18.7%。

### 9.4 ★ 因果矛盾 —— 当天最硬的一条

按位置分桶(bg=0):**0–128 翻 52.85%**,128–256 翻 51.66%,256–512 翻 73.70%,
512–1024 翻 92.75%,1024–2048 翻 86.68%。

**位置 0–128 翻 52.85%,而整条只有 262–330 token 的行整体只翻 ~2%。**

因果模型里位置 50 的 logprob 只依赖 token 0–50,**不可能知道后面还有没有 2000 个 token**。

⟹ **变的不是位置,也不是"误差沿序列/层累积"。是整个 forward 在按不同方式计算,
而跟着变的是「这次 forward 里有多少 token」。**

这指向**按整批 token 决定行为**的路径:MoE 的分组 GEMM / all-to-all 正是这种(分组取决于
整批的路由);注意力不是(逐位置、因果)。两条旁证:

- **笃定档(margin>2 nat)翻 61.8%** —— 末位数值抖动翻不动那种位置,这不是噪声。
- **`bg=4` 反而比 `bg=0` 好一倍。**serve 日志确认 `enable_prefix_caching=False`
  (不是缓存命中)、`Chunked prefill is enabled with max_num_batched_tokens=8192`。
  背景负载分掉预算 ⟹ 长 prompt 被切块 ⟹ 每次 forward 的 token 变少 ⟹ 更稳。

**还没排除**:本轮每行只有一个长度,"行身份"和"序列长度"是同一个变量,
"那四行本来就难"从未被排除。下一轮把同一批行截到 256/512/1024/2048 重复测
(`prefill_noise_sweep.sh` 的 `SEQLENS`),并加 `MAXBATCHTOK=512` 强制分块的一臂。

### 9.5 又一条被自己的数据推翻的

§6 那张单子上再加两条:

- ❌ **「本底翻转率和语料 mismatch 同量级 ⟹ 语料没问题」。**探针第一版直接这么判了。
  错在:翻转率 78.8% 时,语料对比的**参照物本身是垃圾**,这个比较没有意义。
  ⟹ §2 那个「response 段 mismatch 64%」**仍然悬着,不是已结案**。探针已改成在笃定档
  翻转 >5% 或整体 >25% 时**拒绝下结论**。
- ❌ **「没有证据说明这条线曾经是好的」。**说过头了:176 也是 A3,老 pin 在上面跑出过
  gsm8k 96.59% 和 6 小时 0 error —— 那是证据。**要收窄的只有一句:176 上从没量过逐 token
  复现性**(96.59% 是准确率,一个不复现的服务照样能考 96.59%)。

### 9.6 排除清单(截至 2026-09-23)

| 变量 | 状态 |
|---|---|
| HS dumper | 排除(§1 自洽 oracle) |
| `--async-scheduling` | 排除(§7b①) |
| 「阈值 = `index_topk` 512」 | 作废(§7b②) |
| vllm-ascend pin | **排除** —— 老栈 ≡ 新栈 + AllToAll |
| MegaMoe | **不是根因**,但显著加重 |
| prefix cache | 排除(日志 `enable_prefix_caching=False`) |
| CANN 9.2 vs 9.1 | 有影响但**不单向**,样本量下不能定性 |
| **每次 forward 的 token 数** | ★ **头号嫌疑**,§9.4 |
| 机器本身 | 没有第二台机,分不开 |

### 9.7 工具

`examples/ascend_npu_dflash/` 下:`determinism_sweep.sh`(生成层面)、
`prefill_noise_probe.py` + `prefill_noise_sweep.sh`(纯 prefill)、
`positions_report.py`(逐位置 CSV 切片)、`aux_mix_probe.py`(`main_proj` 三层加权)、
`offline_probes.sh`(不占卡的三个一次跑完)。
