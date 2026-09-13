#!/usr/bin/env bash
# Build the LingShu image for one ROLE, staging the copied trees first.
#
# WHY A STAGING STEP. docker 18.09 ships the WHOLE build context to the daemon before the
# first instruction runs. The trees we need live all over the box (conda under one home, CANN
# under another, vLLM under a third) and one of them -- the speculators working tree -- is
# 114 GB of logs. So: hard-link what we want into ./stage (instant, no copy, same filesystem),
# and let .dockerignore exclude everything else.
#
#   ROLE=train bash build_image.sh
#   ROLE=serve bash build_image.sh
#
#   CONDA_ENV   conda env name to bake in     (default depends on ROLE)
#   CANN_SRC    CANN tree to bake in          (default /home/a00652497/CANN/9.0.0.0430)
#   TAG         image tag                     (default dsv4-dspark-<role>:<date>)
set -euo pipefail

ROLE="${ROLE:-train}"
case "$ROLE" in
  train) DEF_ENV=dspark-dsv4-compile ;;
  serve) DEF_ENV=dspark-dsv4-austin  ;;
  *) echo "!! ROLE must be train or serve"; exit 1 ;;
esac
CONDA_ENV="${CONDA_ENV:-$DEF_ENV}"
CONDA_ROOT="${CONDA_ROOT:-/home/n84449292/miniconda3}"
CANN_SRC="${CANN_SRC:-/home/a00652497/CANN/9.0.0.0430}"
TAG="${TAG:-dsv4-dspark-${ROLE}:$(date +%Y%m%d)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/stage"

echo "ROLE=$ROLE  env=$CONDA_ENV  cann=$CANN_SRC  tag=$TAG"
[ -d "$CONDA_ROOT/envs/$CONDA_ENV" ] || { echo "!! no such env: $CONDA_ROOT/envs/$CONDA_ENV"; exit 1; }
[ -d "$CANN_SRC/ascend-toolkit" ]    || { echo "!! no ascend-toolkit under $CANN_SRC"; exit 1; }

rm -rf "$STAGE"; mkdir -p "$STAGE"
# cp -al = hard links: instant and costs no extra disk, but ONLY works on one filesystem.
# Fall back to a real copy across filesystems.
_link() { cp -al "$1" "$2" 2>/dev/null || cp -a "$1" "$2"; }

echo ">>> staging conda env ..."; _link "$CONDA_ROOT/envs/$CONDA_ENV" "$STAGE/conda_env"
echo ">>> staging CANN ...";      _link "$CANN_SRC"                   "$STAGE/cann"
mkdir -p "$STAGE/src"
if [ "$ROLE" = "serve" ]; then
  V="${VLLM_SRC:-/home/a00652497/dspark_austin/installation/vllm-v0.23.0}"
  A="${VA_SRC:-/home/a00652497/dspark_austin/installation/vllm-ascend-v4}"
  [ -d "$V" ] && [ -d "$A" ] || { echo "!! serve needs both $V and $A"; exit 1; }
  echo ">>> staging vLLM trees ..."; _link "$V" "$STAGE/src/vllm"; _link "$A" "$STAGE/src/vllm_ascend"
fi

echo ">>> staged sizes:"; du -sh "$STAGE"/* 2>/dev/null

docker build -t "$TAG" \
  --build-arg ROLE="$ROLE" \
  --build-arg CONDA_PREFIX_PATH="$CONDA_ROOT" \
  --build-arg CANN_HOME="$CANN_SRC" \
  "$HERE"

echo; echo "✅ built $TAG"; docker images "$TAG"
echo; echo ">>> 跑灵枢自查:  bash $HERE/check_image.sh $TAG"
echo ">>> 清理暂存:    rm -rf $STAGE"
