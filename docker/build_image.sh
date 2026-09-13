#!/usr/bin/env bash
# Build the LingShu image, staging every copied tree with hard links first.
#
# WHY A STAGING STEP. docker 18.09 ships the WHOLE build context to the daemon before the
# first instruction runs, and the trees we need live under three different home directories.
# `cp -al` hard-links them into ./stage: instant, and zero extra disk on the same filesystem
# (which matters -- on npu-node80 / has only ~13 GB free while CANN alone is 16 GB).
#
#   ROLE=serve bash build_image.sh          # node80: dspark-dsv4-serving
#   ROLE=train bash build_image.sh          # the other A2: dspark-dsv4-compile
#
# Overrides: BASE_IMAGE CONDA_ROOT CONDA_ENV CANN_SRC TAG SKIP_PKGS
set -uo pipefail

ROLE="${ROLE:-serve}"
CONDA_ROOT="${CONDA_ROOT:-/home/n84449292/miniconda3}"
CANN_SRC="${CANN_SRC:-/home/a00652497/CANN/9.1.0.0627}"
BASE_IMAGE="${BASE_IMAGE:-torchspec_ty:116}"
TAG="${TAG:-dsv4-dspark-${ROLE}:$(date +%Y%m%d)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage"

case "$ROLE" in
  serve) CONDA_ENV="${CONDA_ENV:-dspark-dsv4-serving}" ;;
  train) CONDA_ENV="${CONDA_ENV:-dspark-dsv4-compile}" ;;
  *) echo "!! ROLE must be serve or train"; exit 1 ;;
esac

echo "ROLE=$ROLE  env=$CONDA_ENV  cann=$CANN_SRC  base=$BASE_IMAGE  tag=$TAG"
[ -d "$CONDA_ROOT/envs/$CONDA_ENV" ] || { echo "!! 没有这个环境: $CONDA_ROOT/envs/$CONDA_ENV"; exit 1; }
[ -d "$CANN_SRC/ascend-toolkit" ]    || { echo "!! $CANN_SRC 下没有 ascend-toolkit"; exit 1; }
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
  echo "!! 基础镜像不在本地: $BASE_IMAGE   (这两台都拉不到远端镜像)"
  echo "   可选:"; docker images --format '     {{.Repository}}:{{.Tag}}  {{.Size}}' | head -20; exit 1; }

# ── discover the editable source trees from the env itself, never from memory ───────────
echo ">>> 探测 editable 安装指向的绝对路径 ..."
mapfile -t EDIT < <("$CONDA_ROOT/envs/$CONDA_ENV/bin/python" - <<'PYEOF' 2>/dev/null
import importlib, pathlib
for m in ("vllm", "vllm_ascend", "speculators"):
    try:
        f = getattr(importlib.import_module(m), "__file__", "") or ""
    except Exception:
        continue
    if not f or "site-packages" in f:
        continue
    p = pathlib.Path(f).parent            # .../vllm  |  .../src/speculators
    root = p.parent.parent if p.parent.name == "src" else p.parent
    print(root)
PYEOF
)
[ "${#EDIT[@]}" -gt 0 ] && printf '    %s\n' "${EDIT[@]}" || echo "    (无 editable,只有 site-packages)"

rm -rf "$STAGE"; mkdir -p "$STAGE/src"
_link() { cp -al "$1" "$2" 2>/dev/null || cp -a "$1" "$2"; }

echo ">>> 暂存 conda env ..."; _link "$CONDA_ROOT/envs/$CONDA_ENV" "$STAGE/conda_env"
echo ">>> 暂存 CANN ...";      _link "$CANN_SRC"                   "$STAGE/cann"
: > "$STAGE/src/PATHS"
i=0
for src in "${EDIT[@]}"; do
  [ -d "$src" ] || { echo "  !! 跳过不存在的: $src"; continue; }
  i=$((i+1)); d="t${i}"
  _link "$src" "$STAGE/src/$d" && echo "${d}|${src}" >> "$STAGE/src/PATHS" && echo "  暂存 $src -> $d"
done

echo ">>> 暂存体积:"; du -sh "$STAGE"/* 2>/dev/null
echo ">>> 构建上下文所在盘:"; df -h "$HERE" | tail -1

docker build -t "$TAG" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg ROLE="$ROLE" \
  --build-arg CONDA_ROOT="$CONDA_ROOT" \
  --build-arg CONDA_ENV="$CONDA_ENV" \
  --build-arg CANN_HOME="$CANN_SRC" \
  --build-arg SKIP_PKGS="${SKIP_PKGS:-0}" \
  --build-arg http_proxy="${http_proxy:-}" \
  --build-arg https_proxy="${https_proxy:-}" \
  --build-arg no_proxy="${no_proxy:-localhost,127.0.0.1,.huawei.com}" \
  "$HERE" || { echo "!! 构建失败"; exit 1; }

echo; echo "✅ $TAG"; docker images "$TAG"
echo; echo ">>> 自查:   bash $HERE/check_image.sh $TAG"
echo ">>> 导出:   bash $HERE/export_image.sh $TAG"
echo ">>> 清暂存: rm -rf $STAGE"
