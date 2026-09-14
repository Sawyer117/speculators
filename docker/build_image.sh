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
REPO="$(cd "$HERE/.." && pwd)"
# ⚠️ The stage dir MUST live OUTSIDE the repo. speculators is itself one of the trees being
# staged, so a stage dir inside it makes `cp -al` refuse: "cannot copy a directory into itself".
# Keep it on the SAME filesystem as the sources, though, or the hard links degrade to real
# copies (16 GB of CANN). The stage dir doubles as the build context, so the context contains
# exactly what the image needs and nothing else.
STAGE="${STAGE_DIR:-$(dirname "$REPO")/.lingxu_stage_${ROLE}}"

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
mapfile -t EDIT < <(printf '%s\n' "${EDIT[@]}" | grep '^/' || true)
[ "${#EDIT[@]}" -gt 0 ] && printf '    %s\n' "${EDIT[@]}" || echo "    (无 editable,只有 site-packages)"

# ⚠️ CANN ships read-only directories (0555). `cp -al` hard-links the FILES but creates NEW
# directories inheriting that mode, so a plain `rm -rf` on an old stage dies with hundreds of
# "Permission denied" lines. chmod first -- and note this is safe: only the new directories are
# touched, never the CANN originals (files are hard links; unlinking one leaves the other).
if [ -d "$STAGE" ]; then
  echo ">>> 清理上一次的暂存(先给目录加写权限,CANN 原件不受影响) ..."
  find "$STAGE" -type d -exec chmod u+w {} + 2>/dev/null
  rm -rf "$STAGE"
fi
mkdir -p "$STAGE/src"
cp -f "$HERE/Dockerfile" "$HERE/entrypoint.sh" "$HERE/setup_proxy.sh" "$STAGE/"
echo ">>> 暂存目录(在仓库外):$STAGE"
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

# ── record the git state of every editable tree ────────────────────────────────────────
# ⚠️ These are EDITABLE installs: site-packages holds only a .pth pointing here, so whatever
# is checked out in these directories IS the code the image runs. An uncommitted change goes
# in silently and nothing downstream can tell. Write it down, and refuse nothing -- a dirty
# tree is sometimes exactly what makes the stack work (vllm-ascend-serving is a detached HEAD).
MAN="$STAGE/src/GIT_MANIFEST"
{
  echo "# built $(date -Is) on $(hostname) by $(whoami)"
  echo "# ROLE=$ROLE  CONDA_ENV=$CONDA_ENV  CANN=$CANN_SRC  BASE=$BASE_IMAGE"
  echo
  for src in "${EDIT[@]}"; do
    [ -d "$src" ] || continue
    echo "[$src]"
    if [ -d "$src/.git" ]; then
      echo "  commit : $(git -C "$src" rev-parse HEAD 2>/dev/null)"
      echo "  branch : $(git -C "$src" rev-parse --abbrev-ref HEAD 2>/dev/null)"
      echo "  subject: $(git -C "$src" log -1 --format=%s 2>/dev/null)"
      n=$(git -C "$src" status --porcelain 2>/dev/null | wc -l)
      if [ "$n" != 0 ]; then
        echo "  ⚠ DIRTY: $n uncommitted change(s) baked into this image:"
        git -C "$src" status --porcelain 2>/dev/null | sed 's/^/      /'
      else
        echo "  clean  : yes"
      fi
    else
      echo "  (not a git checkout)"
    fi
    echo
  done
} > "$MAN"
echo ">>> git 状态:"; sed 's/^/    /' "$MAN"

echo ">>> 暂存体积:"; du -sh "$STAGE"/* 2>/dev/null
echo ">>> 构建上下文所在盘:"; df -h "$STAGE" | tail -1

docker build -t "$TAG" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg ROLE="$ROLE" \
  --build-arg CONDA_ROOT="$CONDA_ROOT" \
  --build-arg CONDA_ENV="$CONDA_ENV" \
  --build-arg CANN_HOME="$CANN_SRC" \
  --build-arg SKIP_PKGS="${SKIP_PKGS:-0}" \
  --build-arg BUILD_PROXY="${http_proxy:-}" \
  --build-arg BUILD_NO_PROXY="${no_proxy:-localhost,127.0.0.1,.huawei.com}" \
  "$STAGE" || { echo "!! 构建失败"; exit 1; }

echo; echo "✅ $TAG"; docker images "$TAG"
echo; echo ">>> 自查:   bash $HERE/check_image.sh $TAG"
echo ">>> 导出:   bash $HERE/export_image.sh $TAG"
echo ">>> 清暂存: rm -rf $STAGE"
