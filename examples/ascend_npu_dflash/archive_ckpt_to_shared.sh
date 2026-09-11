#!/usr/bin/env bash
# Move finished checkpoint assets off the local disk to shared storage, leaving a symlink
# behind so every recorded path keeps working.
#
# WHY NOT `mv`. The local disk and the shared mount are DIFFERENT filesystems, so `mv` is a
# copy followed by a delete -- and at TB scale over a network mount an interrupted `mv`
# leaves a PARTIAL copy and has already started removing the source. This script copies,
# VERIFIES (file count + byte count), and only then removes the source.
#
# ⚠️ THE GUARD THAT MATTERS: it refuses any directory that was written to in the last
# $MIN_IDLE_MIN minutes. A training run checkpoints every ~0.5 epoch, so a live run always
# looks recent and can never be archived by accident. Do not "just this once" past it --
# archiving a run mid-checkpoint corrupts the save it is in the middle of writing.
#
# USAGE
#   DRY_RUN=1 bash archive_ckpt_to_shared.sh <dir> [<dir> ...]     # default: show the plan
#   DRY_RUN=0 bash archive_ckpt_to_shared.sh <dir> [<dir> ...]     # actually do it
#
#   DEST=<path>          where the assets go (default below)
#   MIN_IDLE_MIN=<n>     refuse dirs touched more recently than this (default 120)
#   KEEP_SOURCE=1        copy + symlink but do NOT delete the source (needs 2x space)
#
# Re-running is safe: an already-archived dir (source is a symlink) is skipped.
set -uo pipefail

DEST="${DEST:-/share/canada_group_folder/ckpt/dspark_reproduction_result_final_asset}"
DRY_RUN="${DRY_RUN:-1}"
MIN_IDLE_MIN="${MIN_IDLE_MIN:-120}"
KEEP_SOURCE="${KEEP_SOURCE:-0}"

[ $# -ge 1 ] || { echo "usage: [DRY_RUN=0] bash $0 <ckpt-dir> [...]"; exit 1; }

echo "==================================================================="
echo " DEST         = $DEST"
echo " DRY_RUN      = $DRY_RUN   (1 = plan only)"
echo " MIN_IDLE_MIN = $MIN_IDLE_MIN"
echo "==================================================================="

mkdir -p "$DEST" 2>/dev/null || { echo "!! cannot create $DEST"; exit 1; }
[ -w "$DEST" ] || { echo "!! $DEST is not writable"; exit 1; }

# --- space check BEFORE anything is copied -------------------------------------------
need=0
for d in "$@"; do
  [ -d "$d" ] || continue
  [ -L "$d" ] && continue
  need=$(( need + $(du -sk --apparent-size "$d" 2>/dev/null | cut -f1) ))
done
avail=$(df -Pk "$DEST" | awk 'NR==2{print $4}')
printf " need %.1f TB   |   %s has %.1f TB free\n" \
  "$(echo "$need/1073741824" | bc -l)" "$DEST" "$(echo "$avail/1073741824" | bc -l)"
if [ "$need" -gt "$avail" ]; then
  echo "!! not enough room on the destination. Archive fewer dirs, or free space there first."
  exit 1
fi
echo

rc=0
for src in "$@"; do
  name="$(basename "$src")"
  tgt="$DEST/$name"
  echo "───── $name ─────"

  if [ -L "$src" ]; then echo "  skip: already a symlink -> $(readlink "$src")"; continue; fi
  if [ ! -d "$src" ]; then echo "  skip: not a directory"; continue; fi
  if [ -e "$tgt" ]; then echo "  !! destination already exists: $tgt — resolve by hand"; rc=1; continue; fi

  # ⚠️ the live-run guard
  recent=$(find "$src" -mmin "-$MIN_IDLE_MIN" -print -quit 2>/dev/null)
  if [ -n "$recent" ]; then
    echo "  ⛔ REFUSED: written to within ${MIN_IDLE_MIN} min (e.g. $(basename "$recent"))"
    echo "     A live training run checkpoints periodically and must never be archived."
    rc=1; continue
  fi

  n_src=$(find "$src" -type f | wc -l)
  b_src=$(du -sb "$src" | cut -f1)
  printf "  source: %s files, %.1f GB\n" "$n_src" "$(echo "$b_src/1073741824" | bc -l)"

  if [ "$DRY_RUN" = "1" ]; then
    echo "  would: rsync -> $tgt ; verify ; rm source ; ln -s"
    continue
  fi

  echo "  copying ..."
  rsync -a --info=progress2 "$src/" "$tgt/" || { echo "  !! rsync failed — SOURCE UNTOUCHED"; rc=1; continue; }

  n_dst=$(find "$tgt" -type f | wc -l)
  b_dst=$(du -sb "$tgt" | cut -f1)
  if [ "$n_src" != "$n_dst" ] || [ "$b_src" != "$b_dst" ]; then
    echo "  !! VERIFY FAILED  files $n_src/$n_dst  bytes $b_src/$b_dst  — SOURCE UNTOUCHED"
    rc=1; continue
  fi
  echo "  verified: $n_dst files, bytes match exactly"

  if [ "$KEEP_SOURCE" = "1" ]; then
    echo "  KEEP_SOURCE=1 → source kept, no symlink made"
    continue
  fi

  rm -rf "$src" && ln -s "$tgt" "$src" \
    && echo "  ✅ $src -> $tgt" || { echo "  !! failed to swap in the symlink"; rc=1; }
done

echo
echo "剩余空间:"; df -h "$(dirname "${1%/}")" "$DEST" 2>/dev/null | sort -u
exit $rc
