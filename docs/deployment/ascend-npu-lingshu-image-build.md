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

---

## 4. 待办

- [ ] `ROLE=train` 在 **109** 上 build(109 无 DNS ⟹ `SKIP_PKGS=1`,或用本地 `ascend-verl:v4` 作 base)
- [ ] 问平台方:驱动/device 挂载方式、权重卷挂载、环境变量注入、是否要 executor-server、base 镜像有无限制
