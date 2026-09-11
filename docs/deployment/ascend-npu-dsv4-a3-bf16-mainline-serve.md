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
> 相邻文档:A2 w8a8 主线栈的建栈过程 = [`ascend-npu-dsv4-improvement-experiments.md`](./ascend-npu-dsv4-improvement-experiments.md) §18
> (**那份是历史记录,不要改写**);A3 bf16 在**老栈**上的性能数 =
> [`ascend-npu-dsv4-a3-singlenode-benchmark.md`](./ascend-npu-dsv4-a3-singlenode-benchmark.md)。

---

## 状态

| | |
|---|---|
| 当前进度 | **✅ 环境装通**(2026-09-12 02:40)。下一步:起 bf16 服务 |
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
本 fork 是**公开**的。只记脚本路径,不记 URL。推日志前先
`grep -i 'ptaishan\|@90\.' <log>` 核一遍。

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

## 6. 待验证 / 未闭合

| # | 问题 | 状态 |
|---|---|---|
| 1 | **CANN 9.2.0-beta1 + torch-npu 2.10.0.post4 能否编过 V4/SAS 算子** | ✅ **能。2026-09-12 实测装通。** 沿途三个证据点:① `set_env.sh` 正常 source;② 工具链在原位(`lld=…/9.2.0-beta.1/bin/lld`,beta 没挪);③ `torch 2.10.0+cpu \| npu True` —— torch-npu import 成功。脚本第 0 步那句 `⚠ NOT 9.1.0` 只是告警,不是拦截。**⟹ 9.1.0 不是硬下限,9.2.0-beta1 可用。** |
| 2 | `serve_dsv4_a3_singlenode.sh` 是按**老栈**(vLLM 0.23.0 + fork)写的,在 0.27.1 + 上游 main 上旗标可能已挪位或改名 | ⏳ 起服务时逐条过 |
| 3 | bf16 拓扑 = **DP2 × TP8**,不是 w8a8 的 DP4×TP4 | 📌 已知,来自 A3 老栈的实测记录:TP8 → dense/8 ≈ 37 GB 权重 + ~15 GB KV/device;DP4×TP4 对 bf16 太紧 |
| 4 | A3 专属 env 是否仍需要 / 是否仍是这几个 | ⏳ 老栈上是 `ASCEND_A3_ENABLE=1` / `VLLM_ASCEND_ENABLE_FUSED_MC2=1` / `HCCL_BUFFSIZE=1024`,且 MTP method 为 `deepseek_mtp`(非 A2 的 `mtp`) |
| 5 | 是否带 DSpark 草稿做投机,还是先起纯 AR 基线 | ⏳ 未定 |
