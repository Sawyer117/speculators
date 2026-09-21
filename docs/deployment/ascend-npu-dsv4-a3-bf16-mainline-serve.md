# A3 + bf16 + 主线栈 —— DSV4-Flash 推理环境实况记录

> **📝 这是一份实时记录,边做边写。** 做了什么就记什么;发现记错了就改,并把"原来记的是什么、
> 为什么错"留在原地。**没有推测冒充实测** —— 每条都标了是「实测」还是「待验」。
>
> **这是第四种组合,四个维度里三个是新的:**
>
> | 维度 | 已有记录 | 本次 |
> |---|---|---|
> | 机型 | A2(两节点 / 单机 w8a8) | **A3 单机** |
> | 精度 | w8a8 | **bf16** |
> | 栈 | vLLM 0.23.0 + 我们 fork 的 `dspark-dsv4` | **vLLM 0.27.1 + vllm-ascend 上游 main** |
> | CANN | 9.0.0(训练栈)/ 9.1.0.0627(A2 主线栈) | **9.2.0-beta1** ⚠️ 无人验过 |
>
> **★ 2026-09-22:上面那个「无人验过」后来出事了。** 这台机在上下文 ≥512 时温度 0 不可复现
> —— 同一个 prompt 连打三次三个输出,**静默算错而不是崩**。调查全过程、已排除与未排除的
> 变量见 [`ascend-npu-dsv4-a3-186-nondeterminism.md`](ascend-npu-dsv4-a3-186-nondeterminism.md)。
> CANN 9.2.0-beta1 是目前**唯一没被排除**的软件变量。在定性前,这台机产的 HS 不可信。
>
> 相邻文档:A2 w8a8 主线栈的建栈过程 = [`ascend-npu-dsv4-improvement-experiments.md`](./ascend-npu-dsv4-improvement-experiments.md) §18
> (**那份是历史记录,不要改写**);A3 bf16 在**老栈**上的性能数 =
> [`ascend-npu-dsv4-a3-singlenode-benchmark.md`](./ascend-npu-dsv4-a3-singlenode-benchmark.md)。

---

## 状态

| | |
|---|---|
| 当前进度 | **✅ 栈已标定**(2026-09-12)。released draft 五项全量跑完,新栈基准已确立,见 §10 |
| 分支 | `feat/dsv4-dspark-block16` |
| 起始日期 | 2026-09-11 |

---

## 1. 机器与路径

```
开发根目录   /home/a00652497                     ⚠️ 登录账号与开发目录不同名(与 136 那台同一个坑)
安装根 ROOT  /home/a00652497/dsv4_serve
  ├─ speculators/                                feat/dsv4-dspark-block16
  └─ installation/{vllm-v0.27.1, vllm-ascend}    由安装脚本创建
conda        dspark-dsv4-serving  (py3.11, miniforge)
CANN         /data0/canada_group_folder/CANN/9.2.0-beta1/cann-9.2.0-beta.1/
```

⚠️ **共享盘 `canada_group_folder` 在这台上有两个挂载点**:`/home/canada_group_folder` 与
`/data0/canada_group_folder`(用户另做了软链)。**这件事直接影响安装脚本能不能跑起来,见 §3.2。**
连同之前记过的 `/share/` 和 `/mnt/nfs/`,这个共享盘至今出现过 **四个前缀**。

---

## 2. 建 conda 环境(这台没有 conda)

裸机没有任何 conda / mamba / uv。用 miniforge(默认 conda-forge 频道,绕开 Anaconda `defaults`
的 403 许可证门禁 —— 那是 §18 记的坑 #1)。

```bash
cd ~
curl -kfLO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -u -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n dspark-dsv4-serving python=3.11
conda activate dspark-dsv4-serving
python -m pip config set global.trusted-host "pypi.org files.pythonhosted.org mirrors.huaweicloud.com"
```

**⚠️ 这台网络有自签名证书的中间代理。** 三处都会撞:

| 工具 | 症状 | 处置 |
|---|---|---|
| `curl` | `SSL certificate problem: self signed certificate in certificate chain` | `curl -k`(或配企业 CA bundle) |
| `pip` | 同类 SSL 失败 | `pip config set global.trusted-host …` |
| `git clone` | 同类 SSL 失败 | `git -c http.sslVerify=false clone …`(**只对本次生效,不写全局**) |

**★ 这台有现成的代理脚本,不要手填:**

```bash
source /home/a00652497/portproxy_remote.sh     # 设好 http_proxy / https_proxy
export PIP_PROXY="${https_proxy:-$http_proxy}"
export GIT_SSL_NO_VERIFY=1                      # ★ 见下
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}"; export NO_PROXY="$no_proxy"
```

⚠️ **`GIT_SSL_NO_VERIFY=1` 是必须的,而且 `git -c http.sslVerify=false` 不够。**
我们自己 clone speculators 那一条可以用 `-c`,但**安装脚本内部还有两条 git clone**
(第 79 行 vLLM、第 84 行 vllm-ascend),它们跑的是默认配置 ⟹ 走到第 3 步当场挂:

```
fatal: unable to access 'https://github.com/vllm-project/vllm/':
       SSL certificate problem: self signed certificate in certificate chain
```

环境变量对**所有** git 调用生效,这才是对的那把。装完记得 `unset GIT_SSL_NO_VERIFY`。

⚠️ `no_proxy` 里的 `localhost,127.0.0.1` **不是凑数** —— 不设的话起完服务
`curl http://localhost:8000/v1/models` 会被代理劫走,报一个跟服务毫无关系的错。
`quick_serve_check.py`、`/metrics` 轮询、`run_dspark_eval.sh` 全走本地回环。

⚠️ 安装途中会间歇出现 `407 Proxy Authentication Required` 的 WARNING,pip 自己重试后成功,
**不影响结果**,不用管。

⚠️⚠️ **别把代理 URL 写进任何要提交的文件** —— 它是 `user:pass@host:port` 形式,
本 fork 是**公开**的。只记脚本路径,不记 URL。推日志前先用通用模式自查一遍:
```bash
grep -nE '://[^/[:space:]]+:[^/@[:space:]]+@' <log>   # 任何 user:pass@host 形式的 URL
```

有企业 CA bundle 的话更干净,一次配好三者:
`export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem` + `REQUESTS_CA_BUNDLE` +
`CURL_CA_BUNDLE` + `git config --global http.sslCAInfo`。

---

## 3. CANN 9.2.0-beta1

### 3.1 ✅ 9.0.0 那道坎不适用

§18 坑 #4 记的是:`aclmdlRITask was not declared` × 23 —— 真因不是 vllm-ascend 的代码,
而是 **torch-npu `2.10.0.post4` 自带的 `acl_rt.h` 引用了 CANN 9.1.0 才有的类型**。

9.2.0-beta1 上**实测该符号存在**,所以这道坎过了:

```bash
grep -rl aclmdlRITask "$ASCEND_HOME_PATH/include/"     # 有命中
```

⚠️ **但这只排除了一个已知坑,不等于 9.2.0-beta1 可用。** beta 版可能动 ABI、挪
`ccec/bisheng` 工具链、改头文件布局,而这些**只会在第 4 步算子编译(约 40 分钟)时暴露**。
**这是本次唯一没人验证过的变量。**

### 3.2 ★ 混合 CANN 闸门是**逻辑路径字符串前缀**比对 —— 软链接修不了

安装脚本第 82-93 行:

```sh
_cann_root="$(cd "$(dirname "$(dirname "$CANN_ENV")")" && pwd)"   # bash 的 pwd 默认不解软链
case "$ASCEND_HOME_PATH" in "$_cann_root"*) : ;; *) exit 1 ;; esac
```

这台上 `ASCEND_HOME_PATH` 已被预先设成 `/data0/...`,而共享盘同时可从 `/home/...` 访问。
**即使两者指向同一份文件,字符串上也不构成前缀关系 ⇒ 照样 `exit 1`。**

```
❌ CANN_ENV=/home/canada_group_folder/CANN/9.2.0-beta1/ascend-toolkit/set_env.sh
   → _cann_root=/home/...  vs  ASCEND_HOME_PATH=/data0/...     不匹配,退出

✅ CANN_ENV=/data0/canada_group_folder/CANN/9.2.0-beta1/cann-9.2.0-beta.1/set_env.sh
   → _cann_root=/data0/canada_group_folder/CANN/9.2.0-beta1
     ASCEND_HOME_PATH=/data0/canada_group_folder/CANN/9.2.0-beta1/cann-9.2.0-beta.1   前缀匹配 ✓
```

**规则:`CANN_ENV` 必须与已激活的 `ASCEND_HOME_PATH` 落在同一棵目录树下,按字面。**
跑之前先自证:

```bash
_r="$(cd "$(dirname "$(dirname "$CANN_ENV")")" && pwd)"
case "$ASCEND_HOME_PATH" in "$_r"*) echo "✅ 放行" ;; *) echo "❌ 会被拦" ;; esac
```

⚠️ 这道闸不是麻烦,是**省四十分钟的护栏** —— `set_env.sh` 是**前置** PATH 不是替换,
混着两个 CANN 会让旧的 `ccec` 仍排在前面,编译在四十分钟后挂在和之前一模一样的地方,
读起来像"升级没起作用",其实是根本没用上新的。

---

## 4. ★ 安装脚本与量化无关 —— `w8a8` 只是名字

`install_npu_env_dsv4_w8a8.sh` 的名字来自它诞生的那台机器,**脚本本身不含任何量化相关步骤**。
实测核对:`w8a8` 字样只出现在两处纯文本 —— 第 70 行的 banner、第 254 行的 `NEXT:` 提示。

它实际装的:

```
torch 2.10.0 + torch-npu 2.10.0.post4        vLLM v0.27.1 (editable)
vllm-ascend 上游 main @ VA_COMMIT (editable,现编 V4/SAS 算子)
triton-ascend 3.2.2   numpy 2.3.5   torchvision 0.25.0 / torchaudio 2.10.0
+ fastapi / transformers / tokenizers 的 pin 协调(见 §18 的"两处上游 pin 互相矛盾")
```

**量化差异全在起服务的命令里**(`--quantization ascend`、权重路径、拓扑、`max-model-len`、
`gpu_util`),不在环境里。⟹ **bf16 直接用这个脚本,不需要改。**

⚠️ **但脚本末尾打印的 `NEXT: … serve_dsv4_a2_singlenode_w8a8.sh` 对 bf16/A3 是错的,两重错**
(A2≠A3,w8a8≠bf16)。正确的下一步是 `serve_dsv4_a3_singlenode.sh`。

### 4.1 分支选择不是随意的

只有两个分支同时带着**主线安装脚本**和 **A3 起服务脚本**:

| 分支 | `install_npu_env_dsv4_w8a8.sh` | `serve_dsv4_a3_singlenode.sh` |
|---|---|---|
| `main` | ✘ | ✘ |
| `feat/dsv4-dspark` | ✘ | ✔ |
| **`feat/dsv4-dspark-block16`** | **✔** | **✔** |
| `dflash2-reproduce` | ✔ | ✔ |

⟹ 本次用 `feat/dsv4-dspark-block16`。

---

## 5. 安装命令(本次实际执行的)

```bash
CANN_ENV=/data0/canada_group_folder/CANN/9.2.0-beta1/cann-9.2.0-beta.1/set_env.sh
ROOT=/home/a00652497/dsv4_serve

mkdir -p "$ROOT" && cd "$ROOT" && \
git -c http.sslVerify=false clone -b feat/dsv4-dspark-block16 --single-branch \
    https://github.com/Sawyer117/speculators.git && \
cd speculators && \
LOG="$ROOT/install_$(date +%m%d_%H%M).log" && \
CANN_ENV="$CANN_ENV" ROOT="$ROOT" \
  bash examples/ascend_npu_dflash/install_npu_env_dsv4_w8a8.sh 2>&1 | tee "$LOG"
```

- `VA_COMMIT` 用默认 `4ce367a`(#14696 的合入点)。**这一轮只放 CANN 9.2.0-beta1 一个新变量进去** ——
  主线 HEAD 已比它前进约三周,两个未知变量一起上,挂了分不清是谁的锅。想试新的:`VA_COMMIT=main`,
  但**装进另一个 env**。
- 约 40 分钟,大头在编 V4/SAS 算子。**用 tmux 或 nohup,别裸跑在 ssh 里。**
- `ROOT` 需要 **40 GB 以上**空间。

### ✅ 安装结果(2026-09-12 02:40)

```
numpy 2.3.5 | torch 2.10.0+cpu | vllm 0.27.1 | vllm-ascend 4ce367a7d12d
transformers 5.14.1 | tokenizers 0.22.2 | triton-ascend 3.2.2
OK: vLLM + vllm-ascend import cleanly and the ascend platform plugin registers
```

实际落盘路径(**与命令里写的不一样,见下**):

```
/data0/a00652497/dsv4_serve/speculators/
/data0/a00652497/dsv4_serve/installation/vllm-ascend-main/     ← 目录名是 -main,不是 vllm-ascend
/data0/a00652497/dsv4_serve/installation/vllm-v0.27.1/
```

⚠️ **`/home/a00652497` 是指向 `/data0/a00652497` 的软链。** 我们按 `/home/...` 传的 `ROOT`,
自检打印出来是 `/data0/...`。**这是好事** —— `ROOT` 实际落在 3.5T 的 nvme
(`/dev/nvme0n1p1`,2.9T 可用)而不是家目录配额。但**引用路径时要意识到两者等价**,
尤其在又一次遇到字符串比对型的检查时(§3.2 的教训)。

⚠️ 装完打印的 `NEXT: … serve_dsv4_a2_singlenode_w8a8.sh` **对本机是两重错**(A2≠A3,w8a8≠bf16)。
另一行 `NOTE: serve also needs the CANN nnal/atb set_env sourced in a CLEAN shell` 是对的,要照做。

**预期内的依赖冲突**(§18 记过的"两处上游 pin 互相矛盾",保 vLLM 的那一侧):

```
fastapi 0.136.3        vllm-ascend 要 <0.124.0,vLLM 0.27.1 要 >=0.133.0  ⟹ 保 vLLM ✓
transformers 5.14.1    两边都满足 ✓
scipy 1.13.1           被 triton-ascend 钉死,与 numpy 2.3.5 冲突告警 ⟹ 服务路径不用 scipy,留着
numpy                  被 triton-ascend 降到 1.26.4,第 7 步强制拉回 2.3.5 ✓
```

**⏳ 未预期、待观察的缺失依赖** —— vllm-ascend main 声明但没装上的:

```
arctic-inference==0.1.1 · memcache_hybrid==1.2.0 · memfabric_hybrid==1.2.0
pandas-stubs · quart
```

§18 没记过这几个(A2 那次装的是同一个 commit,按理应该一样)。`--no-deps` 装 vllm-ascend
是**有意的**,所以这几个被跳过。**import 自检已经过了**,说明它们不在导入路径上;
但起服务时若报缺模块,先查这张表。

---

## 6. ★ 起服务:四颗雷,全部是老栈没有的

> **这一节是本文档最值钱的部分。** A3 + bf16 在**老栈**(vLLM 0.23.0 + 我们 fork 的
> `386530d12`)上跑通过十几遍(见 [`…-eval-results.md`](./ascend-npu-dsv4-dspark-eval-results.md)
> 里那一长串 `Serve = 176 A3-single`)。换到主线栈后连炸四次,**四颗雷互相独立,一颗都绕不过去。**

⚠️ **先说清楚"以前能现在不能"的三个不同答案**,免得再绕弯路(我们绕了):

| 机器 | 精度 | 栈 | 为什么它没事 |
|---|---|---|---|
| **176**(bf16 eval 跑了十几遍) | bf16 | vLLM 0.23.0 + fork `386530d12` | **老栈,下面四段代码那时都还不存在** |
| **136**(现在仍能 eval) | **w8a8** | vLLM 0.27.1 + main | **双重免疫**:走量化 `quant_method`,根本不进 `routed_experts.py`;而且 A2 脚本**压根没设 `FUSED_MC2`** |
| **本机** | bf16 | 0.27.1 + main | ← 这一格从来没人跑过 |

---

### 雷 1 —— `Failed to load the backend extension: torch_npu`

**症状**:`vllm/env_override.py` 里 `import torch` 就炸,根因是
`/usr/lib64/libstdc++.so.6` 缺 `CXXABI_1.3.15`。

**真因**:conda-forge 的 `libsqlite` 带 ICU 扩展,`import sqlite3` 拉 `libicui18n.so.78`,
它需要比系统更新的 `libstdc++`。环境里**装了**够新的,但链接器先命中系统那个。

**为什么只在起服务时炸**:安装脚本第 125 行有这一行,**A3 起服务脚本没有**:

```
install_npu_env_dsv4_w8a8.sh:125      export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:…"   ✔
serve_dsv4_a2_singlenode_w8a8.sh:149  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:…"   ✔
serve_dsv4_a3_singlenode.sh           （没有）                                        ✘
```

**处置**:父 shell 里 `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`。
脚本内部随后 `source CANN` 会往前插它自己的路径,但那些路径里没有 `libstdc++`,
所以 conda 的仍排在 `/usr/lib64` 前面 —— **实测成立**。
⚠️ **每个新 shell 都要重来**,忘了这条错误会原样回来。

---

### 雷 2 —— `enable_multithread_load does not support safetensors_load_strategy='prefetch'`

A3 脚本第 117-118 行同时给了两个加载选项,**0.23.0 上能共存,0.27.1 把它们变成互斥**:

```sh
[ "$PREFETCH" = "1" ] && LOAD_ARGS=(--safetensors-load-strategy prefetch)
LOAD_ARGS+=(--model-loader-extra-config "{\"enable_multithread_load\":true,…}")
```

**处置**:`PREFETCH=0`(保留 16 线程多线程加载,去掉 prefetch)。脚本自带这个旋钮。

---

### 雷 3 —— ★ OOM 60.5 / 61.27 GiB:**上游已修的 bug,我们的 pin 把它钉住了**

**症状**:target 权重装完(`Loading weights took 10.08s`,46/46 分片),倒在
`vllm_ascend/ops/fused_moe/routed_experts.py:95` 的 `w2_weight_list` 克隆上。
注意是倒在 **w2**,说明 **w13 那份已经克隆成功** —— 即专家权重已经被复制了一份半。

**完整因果链**:

```
A3 脚本第 98 行  VLLM_ASCEND_ENABLE_FUSED_MC2=1          （A3 专属三件套之一）
CANN 9.2.0-beta1 带 cann_ops_transformer 这个 python 包
   ↳ ascend_config.py:27
     _MEGA_MOE_SUPPORTED = importlib.util.find_spec("cann_ops_transformer") is not None  → True
DSV4 过了 _is_megamoe_supported_by_config                → enable_fused_mc2 保持 1
⟹ 走进 _MEGA_MOE_SUPPORTED 分支：w13/w2 各克隆一份
   而该分支【不 del 原张量】【不 empty_cache】—— 相邻的 dynamic_eplb 分支两样都做了
⟹ 本地专家权重驻留 ×2 → OOM
```

⭐ **`apply()` 证明原张量是死重量**(同文件 123-149 行):

```python
w1 = w13_weight_list if isinstance(w13_weight_list, list) else [layer.w13_weight]
```

列表存在时,`layer.w13_weight` 在前向里一次都不会被读。

**处置 = 上游 PR [#15216](https://github.com/vllm-project/vllm-ascend/pull/15216)**
(`e41435157`,*[BugFix] Release unused unquantized weights for fused MC2*,Jade Zheng,2026-08-29 合入)。
作者的描述与我们的诊断一字不差,他给的算例(43 层 MoE / 每 rank 4 专家 / hidden 4096 / 中间维 2048 / bf16)
省 **8.06 GiB**。

```bash
cd <VA_DIR>
git fetch origin main
git show e41435157d3a80937bb01f60efb46b28d663f2d5 -- vllm_ascend/ops/fused_moe/routed_experts.py | git apply -v
# editable 安装,不用重装
```

**实测干净可打**(一个 hunk,offset 1 行),打完专家权重不再翻倍,**mega_moe 快路也保住了**
—— 比 `VLLM_ASCEND_ENABLE_FUSED_MC2=0` 那条绕路强。

⚠️⚠️ **这颗雷的教训比修法本身重要:`VA_COMMIT=4ce367a`(#14696,08-21)这个 pin 有两面性。**
当初钉在那里的理由是"#14696 是 DSpark 能跑的地板";但它**同时把 8 天后(08-29)才修的这个 OOM
一起钉住了**。⟹ **pin 一个 commit 等于同时拒绝了它之后的所有修复**,升级窗口要定期重评。

📌 同一段代码后来还有 [#15303](https://github.com/vllm-project/vllm-ascend/pull/15303)
(`_MEGA_MOE_SUPPORTED` 从模块级 stale import 改成运行时 `use_cann_megamoe()`,修"想关 megamoe 却关不掉")。
**我们不需要**(我们要它开着),但升 `VA_COMMIT` 时会一并拿到。

---

### 雷 4 —— `assert "mtp.0." in name`:`method=mtp` 在主线上不再等价于 `dspark`

**症状**:target 装完、进到 `Loading drafter model...` 后炸在
`vllm_ascend/models/deepseek_v4/mtp.py:301`。

**真因**:那是**单层** MTP 的模型类,`load_weights` 硬断言 `assert "mtp.0." in name`,
后面全是 `name.replace("mtp.0.", "model.layers.0…")`。**DSpark 草稿是 3 层(`mtp.0/1/2`)**,
第一个 `mtp.1.*` 就炸。

**为什么老栈没事**:`serve_dsv4_a3_singlenode_specmethod.sh` 的头注释记着 ——
老栈上 `get_spec_decode_method()` 对两者返回同一个 `AscendDsparkProposer`,
它在 `__init__` 里把 `self.method` 覆写成 `"dflash"`,所以 method 字符串**不影响任何分支**。
**主线上不再如此**:`method` 直接决定加载哪个模型类。

主线里 `dspark` 是一等公民:

```
spec_decode/__init__.py:44        elif method == "dspark":
spec_decode/dspark_proposer.py    专门的 AscendDsparkProposer
llm_base_proposer.py              十余处 self.method == "dspark" 分支
```

**处置**:用 `serve_dsv4_a3_singlenode_specmethod.sh`(它与父脚本**只差 method 一处**,
默认 `SPEC_METHOD=dspark`)。确认:`grep -m1 "Loading draft model" <log>` 应显示 `method=dspark`。

---

### ✅ 起通的完整命令(2026-09-12 03:54)

```bash
source /home/a00652497/portproxy_remote.sh
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}"; export NO_PROXY="$no_proxy"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dspark-dsv4-serving
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"      # ← 雷 1
cd /home/a00652497/dsv4_serve/speculators

CANN_ENV=/data0/canada_group_folder/CANN/9.2.0-beta1/cann-9.2.0-beta.1/set_env.sh \
MODEL=/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 \
DRAFT=/home/canada_group_folder/ckpt/released_draft_bf16_standalone \
NUM_SPEC=5 PREFETCH=0 \                                             # ← 雷 2
  nohup bash examples/ascend_npu_dflash/serve_dsv4_a3_singlenode_specmethod.sh \
  > ~/dsv4_a3_dspark.log 2>&1 &                                      # ← 雷 4
```

(雷 3 是代码补丁,已打在 `<VA_DIR>` 上,不在命令里。)

⚠️ eval 客户端的 `TOKENIZER` 默认指向 `/share/canada_group_folder/...`,
**本机是 `/home/...`,必须覆盖**:

```bash
PORT=7000 DATASET=all CONCURRENCY=48 NUM_PROMPTS=0 KEEP_WARMUP=0 \
TOKENIZER=/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 \
  bash examples/ascend_npu_dflash/run_dspark_eval.sh
```

---

## 7. 给新脚本的差异清单

雷 1/2/4 都是**脚本层面**的,应当固化进一个
`serve_dsv4_a3_singlenode_mainline.sh`(**新开文件,不动仍在给 176 用的那个**):

| 差异 | 老栈 | 主线栈 |
|---|---|---|
| `LD_LIBRARY_PATH` | 不需要 | **必须** `$CONDA_PREFIX/lib` 前置 |
| `PREFETCH` | 1(可与多线程加载共存) | **0**(0.27.1 上互斥) |
| spec `method` | `mtp` ≡ `dspark` | **必须 `dspark`** |
| `--no-disable-hybrid-kv-cache-manager` | 无 | 建议加(`MAXLEN=8192` 下无害;长上下文时 KV 差 3.6×) |

---

## 8. 仍未闭合

| # | 问题 | 状态 |
|---|---|---|
| 1 | released draft 在本栈上的 accept_len | ✅ **已测,见 §10**。统一低约 3%,且缺口在五个数据集和五个位置上都是平的 ⟹ 系统性数值差异,不是配置错。**新栈的横杆 = gsm8k 4.523 / 五项 4.287** |
| 2 | `--additional-config` + `num_spec=5` / TP=8 的 issue #14260 | ✅ 没触发(脚本在有 `DRAFT` 时自动 `FLASHCOMM1=0`) |
| 3 | 主线栈 vs 老栈的吞吐 | 🟡 **新栈明显更快**:gsm8k 843 tok/s vs 176 老栈 conc48 的 590-610。⚠️ 非受控对比(那边是我们的草稿,这边是 released),但 accept_len 接近而吞吐差 40%,大头应是栈。严格结论需同机同权重 A/B |
| 4 | `VA_COMMIT` 要不要从 `4ce367a` 往前推 | ⏳ 等 ① 的数出来再评估;推的话会一并拿到 #15303 |
| 5 | vllm-ascend main 声明但被 `--no-deps` 跳过的五个依赖 | ✅ 起服务没报缺模块,确认不在路径上 |

---

## 9. ★ eval 数据集:代理后面的三连坑(2026-09-12,一次性解决)

`run_dspark_eval.sh` **默认 `OFFLINE=1`,这是对的** —— eval 是几小时的活,跑起来之后不该再依赖网络。
但前提是数据集已经在缓存里。新机器上没有,而在企业 MITM 代理后面把它们弄进来
**连踩三个坑,而且每个报错都指向别处**。

⟹ 已固化成工具:**`examples/ascend_npu_dflash/fetch_eval_datasets.py`**

```bash
source /home/a00652497/portproxy_remote.sh
python examples/ascend_npu_dflash/fetch_eval_datasets.py --insecure
# 有企业 CA 的话用这个更好(校验保持开启):
#   python …/fetch_eval_datasets.py --ca-bundle /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
```

期望 **1319 / 500 / 164 / 257 / 80** 五行 `OK`,然后 eval 照默认 `OFFLINE=1` 跑。

### 三个坑,按触发顺序

| # | 报错 | 真因 | 处置 |
|---|---|---|---|
| 1a | `HTTP Error 504` on every HEAD | `HF_ENDPOINT` 指向华为镜像 `mirrors.tools.huawei.com/huggingface` —— **它不服务 datasets 命名空间**。直连 + 代理才是通的 | `unset HF_ENDPOINT` |
| 1b | `UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol` | ⚠️⚠️ **把 `HF_ENDPOINT` "清空"成 `''` 比不设还糟** —— 空串让 httpx 直接报协议缺失,读起来像 datasets 的 bug | **必须 `unset`,不能设成空** |
| 2 | `File reconstruction error: CAS Client Error` | `huggingface_hub` 默认走 **Xet/CAS 分块后端**,取块的是 `us.aws.cdn.hf.co`(**与 `huggingface.co` 不同的域**)。元数据能过所以拿得到签名 URL,取块失败。**识别特征:进度条先 `downloading bytes: 0.00B` 再 `reconstructing file: 0%`** | `HF_HUB_DISABLE_XET=1` |
| 3 | `[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate in certificate chain` | 代理的自签名证书。⚠️ **`huggingface_hub` ≥1.x 已改用 httpx**,`requests` 时代那些旋钮全部无效 | httpx 认 `SSL_CERT_FILE`(首选);或 monkey-patch `httpx.Client/AsyncClient.__init__` 强制 `verify=False` |

### ⚠️ 一个会骗人的现象

**已经缓存过的数据集在网络完全不通时照样返回 `OK`** —— `datasets` 会静默回落到缓存并打印
`Using the latest cached version`。我们排查时 `MATH-500` 一直 `OK`,差点据此判断"网络是通的"。
**别拿单独一行 OK 当网络正常的证据,要看整张表。**

### 其他

- eval 客户端的 `TOKENIZER` 默认探测 `/share` `/home` `/mnt/nfs` 三个前缀后回落到 `/share/…`,
  **本机在 `/home/…`,必须显式传** `TOKENIZER=/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16`。
- `Warning: You are sending unauthenticated requests to the HF Hub` 不用管,五个小数据集碰不到限流。

---

## 10. ★ 新栈的标定:released draft 五项全量(2026-09-12)

**为什么必须先跑 released**:主线栈换掉了 vLLM、vllm-ascend、CANN 三样东西。
拿老栈的 4.42/4.665 当尺子去量新栈上的草稿,会把栈的差异算到草稿头上。
⟹ **同一份 released 权重,在新栈上重测一遍,得到本栈自己的横杆。**

配置:A3 单机 · bf16 · DP2×TP8×EP16 · `num_spec=5` · conc48 · greedy temp0 ·
`KEEP_WARMUP=0` · OLD set(1309/490/154/247/70)—— 与 ledger 中那批 176 的行对齐。

| 数据集 | 本栈(主线/A3/CANN 9.2.0-beta1) | 老栈 176(`386530d12`) | Δ | 比值 |
|---|---|---|---|---|
| gsm8k | **4.523** | 4.665 | −0.142 | 97.0% |
| math500 | **4.545** | 4.639 | −0.094 | 98.0% |
| humaneval | **4.757** | 4.939 | −0.182 | 96.3% |
| mbpp | **4.450** | 4.526 | −0.076 | 98.3% |
| mt-bench | **3.159** | 3.347 | −0.188 | 94.4% |
| **五项均值** | **4.287** | 4.4232 | −0.136 | **96.9%** |
| **非聊天四项** | **4.569** | 4.6922 | −0.123 | 97.4% |

### ⟹ 本栈的横杆

```
gsm8k 4.523   |   五项均值 4.287   |   非聊天四项 4.569
```

**我们的草稿要比的是这组数,不是 4.42 / 4.665。**

### 为什么判定是"栈差异"而不是"配置错" —— 两个独立角度

**① 跨数据集是平的**:五个全部低 2.0–5.6%,没有单个异常值。配置错通常只打中一类
(比如 chat template 错只伤多轮,block_size 错只伤尾部)。

**② 跨位置也是平的**。把累积逐位反算成条件接受率:

| pos | released c(ledger) | 本栈 c | Δ |
|---|---|---|---|
| 0 | 92.8 | 91.68 | −1.1 pt |
| 1 | 89.2 | 87.97 | −1.2 |
| 2 | 88.5 | 87.44 | −1.1 |
| 3 | 86.8 | 85.12 | −1.7 |
| 4 | 84.2 | 82.38 | −1.8 |

⚠️ **这个形状是判据**:结构性 bug(mask / RoPE / block_size)的特征是
**pos0 几乎不变、缺口随位置急剧扩大** —— 当年 RoPE 那个 degenerate bug 就是这个样子。
这里是**平移**,每一位都低 1.1–1.8 pt,累积起来才成为 accept_len 的 −3%。
⟹ 均匀小数值差,吻合"换了 vLLM / vllm-ascend / CANN 三样东西"。

### 🟡 反向:吞吐明显更高

```
gsm8k 843 tok/s  ·  math500 1230  ·  humaneval 782  ·  mbpp 1453  ·  mt-bench 756
```

176 老栈 conc48 的 gsm8k 一直在 **590–610 tok/s**,这里 **843**,高约 **40%**。

⚠️ **不是受控对比**(ledger 那些吞吐是我们的草稿,这里是 released),所以别当成
精确的加速比。但 accept_len 接近而吞吐差 40%,大头应当是栈——MRV2 + mega_moe 那条快路。
**顺带说明 §6 雷 3 的补丁打对了:省下显存的同时保住了 mega_moe。**

⟹ **主线栈 = accept_len −3%,吞吐 +40%。端到端净赚。**

### 原始日志

`~/eval_released_all_a3main.txt`(A3)。⚠️ 第一次跑在 math500 22% 处被 Ctrl-C,
这张表来自完整的第二次。
