#!/usr/bin/env bash
# Step 6+7 of the LingShu migration guide: export a validated image to a tar, checksum it,
# verify it loads, and print the handoff report.
#
# ⚠️ RUN THE CHECK FIRST. The guide is explicit: "Do not export the final image until all
# mandatory checks pass." An 18 GB tar of a non-compliant image is 20 wasted minutes.
#
#   bash export_image.sh <image:tag> [outdir]
#
# ⚠️ `docker save`, never `docker export` -- export drops image config and history, so the
# result is not a loadable image.
set -uo pipefail

IMG="${1:-}"; OUT="${2:-$HOME/lingxu_export}"
[ -n "$IMG" ] || { echo "usage: bash $0 <image:tag> [outdir]"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "!! no such image: $IMG"; exit 1; }

mkdir -p "$OUT"
NAME="$(echo "$IMG" | tr ':/' '--')"
TAR="$OUT/${NAME}.tar"

SIZE_H=$(docker image inspect "$IMG" --format '{{.Size}}' | awk '{printf "%.1f GB", $1/1073741824}')
AVAIL_K=$(df -Pk "$OUT" | awk 'NR==2{print $4}')
NEED_K=$(docker image inspect "$IMG" --format '{{.Size}}' | awk '{printf "%d", $1/1024}')
echo "image $IMG  ~$SIZE_H   -> $TAR"
if [ "$NEED_K" -gt "$AVAIL_K" ]; then
  echo "!! not enough room in $OUT ($(df -h "$OUT" | awk 'NR==2{print $4}') free) — pick another outdir"
  exit 1
fi

echo ">>> docker save (nohup, survives a dropped terminal) ..."
nohup docker save -o "$TAR" "$IMG" > /tmp/docker_save_${NAME}.log 2>&1 &
SAVE_PID=$!
while kill -0 "$SAVE_PID" 2>/dev/null; do
  printf "\r    %s" "$(ls -lh "$TAR" 2>/dev/null | awk '{print $5}')"; sleep 10
done
echo; wait "$SAVE_PID" || { echo "!! docker save failed — see /tmp/docker_save_${NAME}.log"; exit 1; }

echo ">>> sha256 ..."; SHA=$(sha256sum "$TAR" | cut -d' ' -f1)
echo ">>> verifying the tar loads ..."
docker load -i "$TAR" >/dev/null 2>&1 && echo "    docker load OK" || { echo "!! tar does NOT load"; exit 1; }

cat <<EOF

═══════════════════ 交接信息(照抄给平台) ═══════════════════
Server IP / user : $(hostname -I 2>/dev/null | awk '{print $1}')  /  $(whoami)
Tar path         : $TAR
Image            : $IMG
Size             : $(ls -lh "$TAR" | awk '{print $5}')
SHA256           : $SHA

⚠️ 必须另行迁移(docker commit/save 不含 bind mount 与 volume):
  - 目标模型权重   : <填 DeepSeek-V4-Flash-bf16 的路径>        ~568 GB
  - 草稿权重       : <填 dsv4_dspark_blk15_ep*_vllm-77w>       ~39 GB each
  - 训练数据(Arrow): <填 arrow_0730_77w_dedup>
  - HS dump 目录   : <填 DSPARK_HS_DIR>                        仅训练需要

⚠️ 运行时必须由平台提供(镜像里【故意】没有):
  - /usr/local/Ascend/driver  (只读挂载,驱动与内核绑死)
  - /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
  - 本镜像的 CANN toolkit 需要宿主机驱动 >= 我们验证过的版本
════════════════════════════════════════════════════════════
EOF
