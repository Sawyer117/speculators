#!/usr/bin/env bash
# Verify the DeepSeek-V4-Flash-w8a8-mtp checkpoint against ModelScope's official
# per-file SHA256 manifest — WITHOUT re-downloading (reads local files only).
#
# Source of truth: modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp
#   file-list API (each blob carries a Sha256). Captured into
#   dsv4_w8a8_manifest.sha256 (71 safetensors: 70 weight shards + optional/quarot).
#
# Usage:
#   bash verify_dsv4_w8a8_ckpt.sh [CKPT_DIR]
#     CKPT_DIR defaults to /share/canada_group_folder/ckpt/DeepSeek-V4-Flash-w8a8-mtp
#
# What it does:
#   1. If the ckpt is a git-lfs repo, run `git lfs fsck` (fast, checks content
#      hashes against the recorded LFS OIDs) and stop if that passes.
#   2. Otherwise (or if lfs fsck flags issues) run `sha256sum -c` against the
#      manifest. Reads ~300 GB, so it takes ~10-30 min depending on disk.
#      Any line not ending in ": OK" = a bad or missing shard (re-download just that one).
#
# NOTE: the manifest covers only the 71 *.safetensors weight files. config.json /
# tokenizer / index json are tiny and are NOT a source of NaN, so they're skipped.
set -uo pipefail

CKPT="${1:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-w8a8-mtp}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/dsv4_w8a8_manifest.sha256"

[ -f "$MANIFEST" ] || { echo "!! manifest not found: $MANIFEST"; exit 2; }
[ -d "$CKPT" ]     || { echo "!! ckpt dir not found: $CKPT"; exit 2; }

echo ">>> ckpt     = $CKPT"
echo ">>> manifest = $MANIFEST ($(wc -l < "$MANIFEST") safetensors)"
cd "$CKPT" || exit 2

# --- fast path: git-lfs content check against recorded OIDs ---
if [ -d .git ] && command -v git-lfs >/dev/null 2>&1; then
  echo ">>> git-lfs repo detected — running 'git lfs fsck' (fast path)"
  if git lfs fsck; then
    echo "=== git lfs fsck OK: all LFS objects match their recorded sha256 ==="
    exit 0
  fi
  echo "!! git lfs fsck reported issues — falling through to full sha256 check"
fi

# --- definitive: sha256 every file vs the official manifest ---
echo ">>> sha256sum -c (reads ~300 GB; run under nohup for long jobs)…"
sha256sum -c "$MANIFEST"
rc=$?
echo "=== sha256sum -c exit code: $rc ==="
if [ "$rc" -eq 0 ]; then
  echo "=== ALL 71 SAFETENSORS MATCH — checkpoint is intact ==="
else
  echo "!!! MISMATCH/MISSING — grep the output above for lines NOT ending in ': OK';"
  echo "!!! re-download only those shard(s) from ModelScope."
fi
exit "$rc"
