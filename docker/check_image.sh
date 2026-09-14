#!/bin/bash
# LingShu official requirement check, verbatim in behaviour.
IMAGE=$1
if [ -z "$IMAGE" ]; then echo "Usage: $0 <image>"; exit 1; fi

cid=$(docker create --rm "$IMAGE" 2>/dev/null)
cmd=$(docker inspect --format='{{.Config.Cmd}}' "$cid" 2>/dev/null)
docker rm "$cid" >/dev/null 2>&1
if echo "$cmd" | grep -qE "/bin/bash|sleep"; then
  echo "[PASS] CMD can keep the container running: $cmd"
else
  echo "[FAIL] CMD cannot keep the container running: $cmd"
fi

docker run --rm --entrypoint "" "$IMAGE" /bin/bash -c '
check() { if eval "$2"; then echo "[PASS] $1"; else echo "[FAIL] $1"; fi; }
# --entrypoint "" 跳过了 entrypoint,而这是个非交互 shell ⟹ /root/.bashrc 提前 return。
# 必须显式 source,否则 libhccl.so 找不到,整个栈假 FAIL。
# shellcheck disable=SC1091
[ -f /etc/profile.d/00-dsv4.sh ] && . /etc/profile.d/00-dsv4.sh >/dev/null 2>&1
echo ""; echo "=== Mandatory ==="
check "sshd"        "command -v sshd || test -f /usr/sbin/sshd"
check "sshd_config" "test -f /etc/ssh/sshd_config"
check "chpasswd"    "command -v chpasswd"
echo ""; echo "=== Recommended ==="
check "hostname"    "command -v hostname"
check "SSH host keys generated" "ls /etc/ssh/ssh_host_*_key >/dev/null 2>&1"
if grep -qE "^PermitRootLogin yes" /etc/ssh/sshd_config 2>/dev/null; then
  echo "[PASS] PermitRootLogin yes"; else echo "[WARN] PermitRootLogin is not yes"; fi
check "rsync"       "command -v rsync"
check "dos2unix"    "command -v dos2unix"
echo ""; echo "=== Optional ==="
check "ping"        "command -v ping"
echo ""; echo "=== 镜像里跑的到底是哪份代码(editable 安装 ⟹ 这才是真相) ==="
cat /opt/image_manifest.txt 2>/dev/null || echo "  (无 manifest —— 用旧版 build_image.sh 建的)"
echo ""; echo "=== 我们自己的栈 ==="
check "python 3.11" "python -c \"import sys;assert sys.version_info[:2]==(3,11)\""
check "torch"       "python -c \"import torch\""
check "torch_npu"   "python -c \"import torch_npu\""
# 真实要求(文档 §6 雷 1 实测):conda 的 lib 只要排在【系统 lib 之前】即可 ——
# CANN 的 set_env.sh 会往最前面插它自己的路径,但那些目录里没有 libstdc++,不影响。
# 早先写成 grep "^$CONDA_PREFIX/lib"(必须在第 0 位)比真实要求严,CANN 一插就假 FAIL。
ld_ok() {
  local IFS=: i=0 ci=0 si=0 p
  for p in $LD_LIBRARY_PATH; do
    i=$((i+1))
    [ "$ci" = 0 ] && [ "$p" = "$CONDA_PREFIX/lib" ] && ci=$i
    [ "$si" = 0 ] && { [ "$p" = /usr/lib64 ] || [ "$p" = /usr/lib ] || [ "$p" = /lib64 ]; } && si=$i
  done
  [ "$ci" != 0 ] && { [ "$si" = 0 ] || [ "$ci" -lt "$si" ]; }
}
echo "    LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
check "LD_LIBRARY_PATH: conda lib 早于系统 lib" "ld_ok"
check "CANN set_env 存在" "ls \$ASCEND_TOOLKIT_HOME/../set_env.sh >/dev/null 2>&1 || ls /home/a00652497/CANN/*/ascend-toolkit/set_env.sh >/dev/null 2>&1"
if [ "$ROLE" = "serve" ]; then
  check "vllm"        "python -c \"import vllm\""
  check "vllm_ascend" "python -c \"import vllm_ascend\""
fi
'
