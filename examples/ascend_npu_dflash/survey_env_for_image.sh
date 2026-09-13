#!/usr/bin/env bash
# Survey a box's conda envs / CANN / source trees so a Dockerfile can be written from FACTS.
#
# WHY. Our envs use EDITABLE installs (`pip install -e`), which bind to ABSOLUTE paths via
# .pth files. An image must recreate those exact paths or the install silently breaks. The
# only way to know them is to ask the interpreter where each module actually lives -- version
# numbers alone are not enough.
#
#   bash survey_env_for_image.sh                 # all known envs
#   ENVS="a b" bash survey_env_for_image.sh      # only these
#
# Output is meant to be pasted back verbatim; it prints nothing secret (no proxy URLs, no
# tokens) -- only paths, versions and sizes.
set -uo pipefail

ENVS="${ENVS:-dspark-dsv4-compile dspark-dsv4-serving dspark-dsv4-austin dspark-dsv4-base}"
MODULES="torch torch_npu vllm vllm_ascend speculators transformers numpy triton_ascend safetensors"

_conda() { command -v conda >/dev/null && { conda "$@"; return; }
           for c in ~/miniconda3/bin/conda ~/miniforge3/bin/conda /opt/conda/bin/conda; do
             [ -x "$c" ] && { "$c" "$@"; return; }; done; return 1; }

echo "═══════════ 0. 机器 ═══════════"
echo "host   : $(hostname)"
echo "user   : $(whoami)"
echo "kernel : $(uname -r)   arch: $(uname -m)"
[ -f /etc/os-release ] && . /etc/os-release && echo "os     : ${PRETTY_NAME:-?}"

echo
echo "═══════════ 1. conda 环境 ═══════════"
_conda env list 2>/dev/null || echo "!! 找不到 conda"

echo
echo "═══════════ 2. 每个环境的内容与【绝对路径】 ═══════════"
for E in $ENVS; do
  P=$(_conda env list 2>/dev/null | awk -v e="$E" '$1==e{print $NF}')
  if [ -z "$P" ] || [ ! -x "$P/bin/python" ]; then echo "--- $E : 不存在"; continue; fi
  echo "--- $E"
  echo "    prefix   : $P"
  echo "    env size : $(du -sh "$P" 2>/dev/null | cut -f1)"
  MODULES="$MODULES" "$P/bin/python" - <<'PYEOF' 2>/dev/null
import importlib, os, sys
print("    python   :", sys.version.split()[0])
for m in os.environ["MODULES"].split():
    try:
        x = importlib.import_module(m)
        f = getattr(x, "__file__", "") or ""
        # an editable install lives OUTSIDE site-packages -- that is the path an image must recreate
        tag = "  <== EDITABLE(路径必须在镜像里重建)" if f and "site-packages" not in f else ""
        print(f"    {m:14} {getattr(x,'__version__','?'):26} {f}{tag}")
    except Exception as exc:
        print(f"    {m:14} -- ({type(exc).__name__})")
PYEOF
done

echo
echo "═══════════ 3. CANN 与驱动 ═══════════"
echo "--- toolkit 候选 ---"
ls -d /usr/local/Ascend/ascend-toolkit/*/ ~/CANN/*/ /home/*/CANN/*/ 2>/dev/null | sort -u
echo "--- 体积 ---"
du -sh /usr/local/Ascend/ascend-toolkit ~/CANN/* /home/*/CANN/* 2>/dev/null | sort -u
echo "--- set_env.sh 位置 ---"
find /usr/local/Ascend ~/CANN /home/*/CANN -maxdepth 4 -name set_env.sh 2>/dev/null | head -10
echo "--- 驱动(不进镜像,但版本是硬约束) ---"
cat /usr/local/Ascend/driver/version.info 2>/dev/null | head -5 || echo "  (读不到)"
echo "--- 当前 shell 激活的 ---"
echo "  ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<未设置>}"

echo
echo "═══════════ 4. 源码树 ═══════════"
for d in /home/a00652497/dspark_austin/installation/* \
         /home/a00652497/dspark_2026/installation/* \
         /home/a00652497/dsv4_serve/installation/* \
         /home/a00652497/dspark_austin/speculators \
         /home/a00652497/dsv4_serve/speculators; do
  [ -d "$d" ] || continue
  b=""; [ -d "$d/.git" ] && b="  git=$(git -C "$d" rev-parse --short HEAD 2>/dev/null) $(git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  printf "  %-62s %6s%s\n" "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)" "$b"
done

echo
echo "═══════════ 5. docker 与空间 ═══════════"
docker --version 2>/dev/null || echo "  !! 无 docker"
docker info --format '  storage driver: {{.Driver}}   root: {{.DockerRootDir}}' 2>/dev/null
df -h / /var/lib/docker /data0 2>/dev/null | sort -u
echo "--- 已有镜像(前 5) ---"
docker images 2>/dev/null | head -6

echo
echo "═══════════ 6. 灵枢检查项在【宿主机】上的现状 ═══════════"
for c in sshd rsync dos2unix hostname ping chpasswd; do
  printf "  %-10s %s\n" "$c" "$(command -v $c 2>/dev/null || echo '缺(镜像里要装)')"
done
