# LingShu 镜像制作:DSV4-DSpark serve / train 双角色

> 实时记录。做了什么就记什么;后面发现错了再回来改。
> 脚本:`docker/`(`build_image.sh` / `check_image.sh` / `export_image.sh` / `Dockerfile` / `entrypoint.sh` / `setup_proxy.sh`)

---

## 1. 设计原则:镜像不负责可复现性

**代码必须可变。** 容器里那三个 repo 是**真的 git 工作区**(editable 安装),进去就能 `git fetch && git checkout`。
可复现性由**维护者记录的 commit id** 唯一确定 —— 镜像里烤一份 `/opt/image_manifest.txt`
(`build_image.sh` 生成的 `GIT_MANIFEST`),写明每个 repo 的 commit / branch / subject / 是否 dirty。

反过来做(镜像即真理、代码只读)的后果:改一行就得重新 build 一个 38 GB 的镜像,直接 GG。

## 2. 一次 build 的全貌

```bash
source /home/a00652497/portproxy_remote_linux.sh      # 黄区代理
sudo sysctl -w net.ipv4.ip_forward=1
ROLE=serve bash docker/build_image.sh 2>&1 | tee /tmp/build_serve.log
bash docker/check_image.sh  dsv4-dspark-serve:<DATE>
bash docker/export_image.sh dsv4-dspark-serve:<DATE> /data1/lingxu_export
```

`build_image.sh` 依次做:问目标 conda env 的 python 要 editable 路径 → `cp -al` 硬链接暂存
conda env / CANN / 三个 repo → 写 `GIT_MANIFEST` → `docker build`。

| ROLE | conda env | 机器 | 备注 |
|---|---|---|---|
| `serve` | `dspark-dsv4-serving` | 116 | vLLM + vllm-ascend + speculators |
| `train` | `dspark-dsv4-compile` | 109 | 109 无 DNS ⟹ 需本地 base + `SKIP_PKGS=1` |

**2026-09-14,116,`ROLE=serve` 成功:**`dsv4-dspark-serve:20260914`,38.7 GB,LingShu 的
Mandatory / Recommended / Optional 全 PASS。烤进去的 commit:

```
vllm-v0.23.0            0fc695fc  (Cap fastapi < 0.137 …)
vllm-ascend-serving     386530d1  (DSpark drafter ACLGraph)
speculators             13ae97a0  feat/dsv4-dspark  ⚠ DIRTY: docker/ 那 8 个文件
```

---

## 3. 踩过的坑(每一条都真实发生过)

### 3.1 base 镜像的 `ENV` 会**遮蔽**同名 `ARG`

`--build-arg http_proxy=…` 被静默忽略,apt 仍去连 base 镜像里那个早已失效的代理,
`E: Failed to fetch … Could not connect to 90.255.89.236:3128`,然后 fallback 到
`yum` → `/bin/sh: 1: yum: not found` → `non-zero code: 127`。

**修**:代理参数改名 `BUILD_PROXY` / `BUILD_NO_PROXY`,在 `RUN` 内部 `export` 成 `http_proxy`。
成功的标志是这一行:`>>> apt 走的代理: http://***(已覆盖 base 镜像里那个死的)`。

### 3.2 暂存目录不能放在被暂存的 repo 里

`docker/stage/` 在 speculators 里,而 speculators 本身要被 `cp -al` 进 stage
⟹ `cp: cannot copy a directory into itself`。**修**:默认 `$(dirname $REPO)/.lingxu_stage_${ROLE}`。

### 3.3 清理旧暂存时刷屏 `Permission denied`

CANN 的目录是 **0555 只读**;`cp -al` 硬链接**文件**但**新建目录**并继承该权限
⟹ 目录不可写 ⟹ 里面的条目删不掉。

**修**:删之前 `find "$STAGE" -type d -exec chmod u+w {} +`。
脚本末尾打印的「清暂存」提示也必须带上这个 `chmod`,否则你照着复制一样删不动(2026-09-14 就这么二次踩到)。**手动清理的正确写法**:

```bash
find <STAGE> -type d -exec chmod u+w {} + 2>/dev/null; rm -rf <STAGE>
```
★ 这是安全的:`chmod` 只动新建的副本目录,碰不到 CANN 原件;`rm` 一个硬链接只是减引用计数。

### 3.4 `CMD ["/bin/bash"]` —— 官方自查**会过**,实际**起不来**

没有 TTY 时 bash 立刻退出,容器死掉,Clab exec 不进去。
而官方检查脚本是 `grep -qE "/bin/bash|sleep"`,所以它 **PASS 了却没用**。
**修**:`CMD ["sleep", "infinity"]`。

### 3.5 ★ 环境写进 `~/.bashrc` 对非交互 shell **完全无效**

Ubuntu 的 `/root/.bashrc` 第一行:

```bash
case $- in *i*) ;; *) return;; esac
```

非交互直接 `return` ⟹ 追加在**末尾**的 `source set_env.sh` 一行都不跑 ⟹
`ImportError: libhccl.so: cannot open shared object file`,`import torch` / `vllm` / `vllm_ascend` 全 FAIL。
ssh 进去是交互的,反而看不出问题 —— 所以这坑只在 `docker exec <c> python …` 这类调用里炸。

**修**:环境统一写 `/etc/profile.d/00-dsv4.sh`(`bash -lc` 经 `/etc/profile` 会加载),
`.bashrc` 只留一行 `source` 它,`entrypoint.sh` 也显式 source 一次,
`check_image.sh` 因为用了 `--entrypoint ""` 同样要显式 source。

### 3.6 `pip install -e` 绑的是**绝对路径**

`.pth` 里写死了源码目录,所以镜像必须把这些路径**原样重建**,否则 `import vllm_ascend` 直接炸。
`build_image.sh` 的做法:问目标 env 的 python 要路径 → 暂存成 `t1/t2/t3` → 用 `PATHS` 清单在
镜像里 `mv` 回原位。本次探到三条:

```
/home/a00652497/dspark_austin/installation/vllm-v0.23.0
/home/a00652497/dspark_2026/installation/vllm-ascend-serving
/home/a00652497/dspark_austin/speculators
```

### 3.7 ★ CANN 里的**绝对路径符号链接**指向树外 ⟹ 断链,且 build 不报错

116 上 `/home/a00652497/CANN/9.1.0.0627` 是 `/home/canada_group_account/CANN/9.1.0.0627` 的
`cp -al` 硬链副本(每个条目 link count = 2),而 `ascend-toolkit/{latest,set_env.sh}` 和 `cann`
三条链接**指向 canada_group_account 那棵树**。

`cp -al` 的 `-a` 含 `-d`(不跟随链接)⟹ 链接原样进镜像 ⟹ 目标路径不在镜像里 ⟹ 断链。
**docker build 全程不报错**,16G 也确实搬进去了,四个 `set_env.sh` 也都在,
但要到容器里 `import torch` 才炸:

```
ImportError: libhccl.so: cannot open shared object file
```

**修**:`CANN_SRC` 指向**真正拥有文件的那棵树**(canada_group_account),它内部自洽
(绝对链接全指向自己,其余是相对链接)。`build_image.sh` 现在会在 build 前用
`readlink -f $CANN_SRC/ascend-toolkit/set_env.sh` 校验解析结果是否落在 `$CANN_SRC` 内,
不在就直接 exit 1 并打印越界的链接 —— 不再留到 `check_image.sh` 才发现。

另给老路径 `/home/a00652497/CANN/9.1.0.0627` 补了一条软链(`CANN_ALIAS`),因为
文档和 serve 脚本里写死的是它。

**正确的 build 命令(116):**

```bash
ROLE=serve CANN_SRC=/home/canada_group_account/CANN/9.1.0.0627 bash docker/build_image.sh
```

不用 `cp -aL`(全量跟随):`cann` 和 `cann-9.1.0-beta.3` 本是同一份的两个名字,
跟随会实打实复制两遍,镜像从 38G 涨到 50G+。

### 3.8 自查脚本自己写错的断言

`LD_LIBRARY_PATH` 那条原本写成 `grep -q "^$CONDA_PREFIX/lib"`,要求 conda lib 在**第 0 位**。
但 §6 雷 1 实测下来的真实要求是「conda lib 排在**系统 lib 之前**」—— CANN 的 `set_env.sh`
会往最前面插自己的路径,那些目录里没有 `libstdc++`,不影响。CANN 一插,这条就假 FAIL。
已改成按位置比较。

教训:**自查脚本的断言比被查对象更容易错**。一条 FAIL 先问「断言对不对」,再问「镜像对不对」。

---

## 4. 状态

**2026-09-14 116 `ROLE=serve` 全绿**(`fed102ba36dc`,38.7 GB):

```
Mandatory    sshd / sshd_config / chpasswd                         PASS
Recommended  hostname / host keys / PermitRootLogin / rsync / dos2unix  PASS
Optional     ping                                                  PASS
我们的栈     python3.11 / torch / torch_npu / vllm / vllm_ascend    PASS
```

`can not use command: npu-smi info` 是没挂 NPU 设备时 torch_npu 的正常提示,不是错误。
`libtinfo.so.6: no version information available` 同理 —— conda 的 libtinfo 比系统 bash
编译时用的旧,只是 warning。

## 5. 待办

- [ ] `ROLE=train` 在 **109** 上 build(109 无 DNS ⟹ `SKIP_PKGS=1`,或用本地 `ascend-verl:v4` 作 base)
- [ ] 问平台方:驱动/device 挂载方式、权重卷挂载、环境变量注入、是否要 executor-server、base 镜像有无限制
