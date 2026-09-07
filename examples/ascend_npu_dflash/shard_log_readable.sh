#!/usr/bin/env bash
# Split a run log into READABLE .log shards and push them one per commit.
#
# WHY, NEXT TO archive_log_push.sh. That script ships a log as `part.NNNN.xz` -- compact
# (~20x) and byte-verifiable, but you have to clone and decompress before you can read a
# line. These shards are the plain text: open one in the GitHub UI and read it. Both are
# worth having, and the .xz set stays the authority because it carries the sha256.
#
# ⚠ THE ~100 KB PUSH CAP IS THE TRAINING BOX'S NETWORK, NOT GITHUB AND NOT EVERYWHERE.
# On that box `git push` sends the whole pack as one POST and a 192 KB pack came back
# `HTTP 403`, which is why `push` here commits one shard at a time. Measured from an
# unrestricted machine, 73 shards -- 84 MB of raw log -- went in a SINGLE push in 6.4 s.
# So: if you can reach GitHub without that gateway, skip `push` entirely and just
#     git add <dest_dir> && git commit && git push
# GitHub's own limit is 100 MB per FILE, which nothing here comes close to.
#
# The ~1.2 MB shard size is kept for a different reason: readability. GitHub's blob viewer
# gives up on very large text files, so a 95 MB log would be un-openable in the browser --
# which defeats the point of shipping plain text. 1.2 MB opens instantly, and the split
# only ever falls on a record boundary.
#
# Cuts only at a record start (`^[`), so no shard begins mid-record: the rich logger wraps
# one step across ~26 lines and a byte-split would tear it.
#
# Resumable: shards already tracked by git are skipped, so re-run after any failure.
#
# USAGE
#   shard_log_readable.sh split <logfile> <dest_dir> [name]
#   shard_log_readable.sh push  <dest_dir>
#
# ENV  RAW_TARGET default 1200000 · PACK_LIMIT default 85000 (post-zlib budget per shard)
set -eo pipefail

RAW_TARGET="${RAW_TARGET:-1200000}"
PACK_LIMIT="${PACK_LIMIT:-85000}"
die() { echo "!! $*" >&2; exit 1; }

cmd_split() {
  local log="${1:?usage: split <logfile> <dest_dir> [name]}"
  local dest="${2:?usage: split <logfile> <dest_dir> [name]}"
  local name="${3:-$(basename "${log%.log}")}"
  [ -f "$log" ] || die "no such log: $log"
  case "$dest" in *.log) die "dest dir must not end in .log — .gitignore matches it";; esac

  # A public fork: refuse a log that still carries the box account id. redact_log.py first.
  if grep -qE '/home/[a-z][0-9]{6,}' "$log"; then
    die "$log still has an account id in its paths — run redact_log.py first"
  fi

  mkdir -p "$dest"
  RAW_TARGET="$RAW_TARGET" PACK_LIMIT="$PACK_LIMIT" NAME="$name" \
    python3 - "$log" "$dest" <<'PYSPLIT'
import gzip, os, re, sys
log, dest = sys.argv[1], sys.argv[2]
target, limit, name = (int(os.environ["RAW_TARGET"]), int(os.environ["PACK_LIMIT"]),
                       os.environ["NAME"])
START = re.compile(rb"^\[")
buf, size, idx, worst = [], 0, 1, 0

def flush():
    global buf, size, idx, worst
    if not buf:
        return
    blob = b"".join(buf)
    # git packs with zlib, so that -- not xz -- is what decides whether the push fits.
    worst = max(worst, len(gzip.compress(blob, 9)))
    with open(os.path.join(dest, f"{name}_{idx:03d}.log"), "wb") as fh:
        fh.write(blob)
    idx += 1
    buf, size = [], 0

with open(log, "rb") as fh:
    for line in fh:
        if size >= target and START.match(line):
            flush()
        buf.append(line)
        size += len(line)
flush()
print(f"   {idx - 1} 片 · 最大一片 zlib 后 {worst:,} 字节", end=" ")
print("(在上限内)" if worst <= limit else f"(★超过 {limit:,},调小 RAW_TARGET 重跑)")
sys.exit(0 if worst <= limit else 3)
PYSPLIT
  echo "== 完成 -> $dest  ($(find "$dest" -name '*.log' | wc -l) 片)"
}

cmd_push() {
  local dest="${1:?usage: push <dest_dir>}"
  local branch; branch="$(git rev-parse --abbrev-ref HEAD)"
  local blocked
  blocked="$(git check-ignore $(find "$dest" -name '*.log' | sort) 2>/dev/null || true)"
  [ -z "$blocked" ] || { echo "!! .gitignore 挡住:" >&2; git check-ignore -v $blocked >&2; exit 5; }

  # Drain anything committed but not pushed first, or the next push carries two shards in
  # one request -- exactly what the per-shard split exists to avoid.
  if [ -n "$(git rev-list @{u}..HEAD 2>/dev/null)" ]; then
    git push -q origin "HEAD:$branch" || die "先把已有提交推掉再继续"
  fi

  local f n=0
  for f in $(find "$dest" -name '*.log' | sort); do
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 && continue
    git add "$f"
    git -c user.name='Sawyer117' -c user.email='wensyaustin@foxmail.com' \
        commit -q -m "logs(raw): $(basename "$f")"
    git push -q origin "HEAD:$branch" || {
      echo >&2; echo "!! push 失败于 $f —— 重跑本命令会从这一片继续" >&2; exit 1; }
    n=$((n + 1)); printf '.'
    [ $((n % 50)) -eq 0 ] && echo " $n"
  done
  echo; echo "== 推送完成,本次 $n 片"
}

case "${1:-}" in
  split) shift; cmd_split "$@" ;;
  push)  shift; cmd_push  "$@" ;;
  *) sed -n '/^# USAGE/,/^set -eo/p' "$0" >&2; exit 2 ;;
esac
