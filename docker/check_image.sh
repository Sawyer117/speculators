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
echo ""; echo "=== 我们自己的栈 ==="
check "python 3.11" "python -c \"import sys;assert sys.version_info[:2]==(3,11)\""
check "torch"       "python -c \"import torch\""
check "torch_npu"   "python -c \"import torch_npu\""
check "LD_LIBRARY_PATH 前置 conda lib" "echo \$LD_LIBRARY_PATH | grep -q \"^\$CONDA_PREFIX/lib\""
check "CANN set_env 存在" "ls \$ASCEND_TOOLKIT_HOME/../set_env.sh >/dev/null 2>&1 || ls /home/a00652497/CANN/*/ascend-toolkit/set_env.sh >/dev/null 2>&1"
if [ "$ROLE" = "serve" ]; then
  check "vllm"        "python -c \"import vllm\""
  check "vllm_ascend" "python -c \"import vllm_ascend\""
fi
'
