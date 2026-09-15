#!/usr/bin/env bash
# 读出镜像命名规范需要的字段:python / torch / torch_npu / vllm / CANN / arch / OS。
#
# 为什么需要这个:命名规范要的是【镜像里实际装了什么】,而这些既不在 tar 文件名里,也不在
# 镜像的 config 里 —— 只有进去 import 一次才知道。猜一个版本号写进镜像名,比名字含糊更糟:
# 含糊的名字只是没信息,错的版本号是负信息。
#
# 用法:
#   bash probe_image.sh <image:tag>        # 已经 docker load 过
#   bash probe_image.sh /path/to/x.tar     # 会先 docker load(18 GB 的 tar 要几分钟 + 同等磁盘)
#   FAST=1 bash probe_image.sh x.tar       # 只读 tar 里的 config(免 load,但拿不到包版本)
set -u
SRC="${1:?用法: bash probe_image.sh <image:tag | *.tar>}"

if [ "${FAST:-0}" = "1" ] && [ -f "$SRC" ]; then
  echo ">>> FAST:只读 tar 内的镜像 config(拿不到包版本,只有架构/OS/环境变量)"
  _cfg="$(tar -tf "$SRC" | grep -E '^[0-9a-f]{64}\.json$' | head -1)"
  tar -xOf "$SRC" "$_cfg" | python3 -c '
import json,sys
c=json.load(sys.stdin)
print("architecture:", c.get("architecture"))
print("os          :", c.get("os"))
print("created     :", c.get("created"))
for e in (c.get("config") or {}).get("Env") or []:
    if any(k in e for k in ("PATH=","CONDA","ASCEND","LD_LIBRARY")): print("env         :", e[:160])
'
  exit 0
fi

IMG="$SRC"
if [ -f "$SRC" ]; then
  echo ">>> docker load -i $SRC  (大 tar 要几分钟)"
  IMG="$(docker load -i "$SRC" | sed -n 's/^Loaded image: //p' | tail -1)"
  [ -n "$IMG" ] || { echo "!! docker load 没报出镜像名,手动 docker images 找一下"; exit 1; }
fi
echo ">>> image: $IMG"
docker image inspect -f '    architecture : {{.Architecture}}
    os           : {{.Os}}
    created      : {{.Created}}
    size         : {{.Size}}' "$IMG"

# --entrypoint "" 是必须的:业务镜像的 entrypoint 会直接把服务拉起来。
# bash -lc 也是必须的:环境接线在 /etc/profile.d/,非登录 shell 读不到 -> import torch 直接炸。
docker run --rm --entrypoint "" "$IMG" bash -lc '
  echo "    uname -m     : $(uname -m)"
  . /etc/os-release 2>/dev/null && echo "    os-release   : ${ID}${VERSION_ID}"
  echo "    python       : $(python -V 2>&1 | head -1)"
  python - <<PY 2>&1 | sed "s/^/    /"
import importlib
for m in ("torch", "torch_npu", "vllm", "vllm_ascend", "transformers", "numpy"):
    try:
        v = getattr(importlib.import_module(m), "__version__", "<no __version__>")
    except Exception as exc:
        v = f"<不可用: {type(exc).__name__}>"
    print(f"{m:<13}: {v}")
PY
  _c="${ASCEND_HOME_PATH:-}"
  if [ -n "$_c" ] && [ -f "$_c/version.cfg" ]; then
    echo "    CANN         : $(tr -d " \n" < "$_c/version.cfg")   ($_c)"
  else
    echo "    CANN         : 未 source(ASCEND_HOME_PATH 为空);候选路径:"
    ls -d /usr/local/Ascend/ascend-toolkit/* /home/*/CANN/* 2>/dev/null | sed "s/^/                   /" | head -5
  fi
' || echo "!! 容器内探测失败 —— 该镜像可能没有 bash / python 不在 PATH,手动 docker run -it 进去看"
