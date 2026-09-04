#!/usr/bin/env bash
# Push a big training log through a gateway that caps a single HTTP request body.
#
# WHY THIS EXISTS. A DSV4-DSpark run log is ~253 MB (124k steps x ~26 wrapped lines).
# Compressed it is ~11 MB (xz) / ~18 MB (gzip) -- GitHub takes either (its per-file limit
# is 100 MB) but our gateway does not: it caps ONE request at ~100 KB. `git push` sends
# the whole pack as a single POST, so the cap lands on the PUSH, not on the file.
# Splitting the file is therefore not enough -- each push must also carry less than the
# cap. Hence: one part per commit, one push per commit, resumable.
#
# WHY RECORD BOUNDARIES, NOT BYTES. Each part is cut at a line starting with `[`, i.e. at
# a `[HH:MM:SS]` or `[MOE-LOAD Lx]` record start, and compressed on its own. So every part
# is a readable log fragment you can open directly -- not a binary shard that means
# nothing until the whole set is reassembled.
#
# ⚠ Do NOT "distill" the log by grepping one metric name. The logger wraps a single step
# across ~26 physical lines; `grep global_step=` keeps the one line carrying it and
# silently drops train/loss, accept_len, step_ms and the rest. Measured: 124,480 -> 0.
#
# ⚠ The ORIGINAL log is never written to, moved or deleted by any subcommand here.
#
# ⚠ The destination repo is PUBLIC and these logs carry the box account name in every
# absolute path. `pack` REFUSES a log that still has one -- redact first:
#   python3 redact_log.py run.log run.redacted.log     (and pack the redacted file)
#
# USAGE
#   archive_log_push.sh pack   <logfile> [dest_dir]   # split at records + compress + manifest
#   archive_log_push.sh push   <dest_dir>             # commit+push one part at a time (resumable)
#   archive_log_push.sh verify <dest_dir>             # reassemble, compare sha256 with the original
#   archive_log_push.sh clean  <dest_dir> --yes       # drop the parts from HEAD (history keeps them)
#
# ENV
#   PART_BYTES        default 90000  -- the per-push cap to stay under
#   RAW_TARGET        default 1800000 -- uncompressed bytes per part (~23x -> ~78 KB)
#   COMPRESS          xz | gzip      -- default: xz if present, else gzip
#   ALLOW_UNREDACTED  1 to skip the account-id check (private remote / already clean)
set -euo pipefail

PART_BYTES="${PART_BYTES:-90000}"
RAW_TARGET="${RAW_TARGET:-1800000}"
COMPRESS="${COMPRESS:-$(command -v xz >/dev/null 2>&1 && echo xz || echo gzip)}"

die() { echo "!! $*" >&2; exit 1; }

# Both xz and gzip decode a CONCATENATION of independent streams as one stream, which is
# what makes `cat part.*` -> decompress work even though each part was compressed alone.
ext() { [ "$COMPRESS" = "xz" ] && echo "xz" || echo "gz"; }

cmd_pack() {
  local log="${1:?usage: pack <logfile> [dest_dir]}"
  [ -f "$log" ] || die "no such file: $log"

  # Sawyer117/speculators is PUBLIC and these logs are full of absolute paths carrying the
  # box account name (/home/a00652497/..., the conda env under /home/n84449292/...).
  # Refuse rather than trust anyone to remember: once a part is pushed it is in history.
  # ALLOW_UNREDACTED=1 is the escape hatch for a genuinely clean log or a private remote.
  if [ "${ALLOW_UNREDACTED:-0}" != "1" ]; then
    local here ids
    here="$(dirname "$0")"
    if ids="$(python3 "$here/redact_log.py" --scan "$log")"; then
      : # clean
    else
      echo "!! 拒绝打包:日志里有账号 ID($ids)—— 目标仓库是公开的。" >&2
      echo "   先脱敏,再对脱敏后的文件打包:" >&2
      echo "     python3 $here/redact_log.py '$log' '${log%.log}.redacted.log'" >&2
      echo "     $0 pack '${log%.log}.redacted.log'" >&2
      echo "   (确实无需脱敏时:ALLOW_UNREDACTED=1 $0 pack ...)" >&2
      exit 4
    fi
  fi

  local name dest e
  name="$(basename "$log")"
  # Strip the trailing .log when naming the DIRECTORY: .gitignore carries a bare `*.log`,
  # which matches directories too, so `docs/deployment/logs/run.redacted.log/` was silently
  # unaddable and `push` died at `git add` with "paths are ignored by one of your
  # .gitignore files". MANIFEST still records the full original name, so verify is
  # unaffected. (The first archive escaped this only by being a single .xz file.)
  dest="${2:-docs/deployment/logs/${name%.log}}"
  e="$(ext)"
  mkdir -p "$dest"

  local raw_sha raw_size
  raw_sha="$(sha256sum "$log" | cut -d' ' -f1)"
  raw_size="$(stat -c%s "$log")"

  echo "== 切片 + 压缩 ($COMPRESS, 每片约 $RAW_TARGET 未压缩字节, 只在记录边界落刀)"
  rm -f "$dest"/part.*
  COMPRESS="$COMPRESS" RAW_TARGET="$RAW_TARGET" PART_BYTES="$PART_BYTES" \
    python3 - "$log" "$dest" "$e" <<'PYSPLIT'
import gzip, lzma, os, re, sys
log, dest, e = sys.argv[1], sys.argv[2], sys.argv[3]
target = int(os.environ["RAW_TARGET"])
limit = int(os.environ["PART_BYTES"])
comp = (lambda b: lzma.compress(b, preset=9)) if os.environ["COMPRESS"] == "xz" \
    else (lambda b: gzip.compress(b, 9))
START = re.compile(rb"^\[")            # [HH:MM:SS] ... or [MOE-LOAD Lx] ...
buf, size, idx, worst, total = [], 0, 0, 0, 0

def flush():
    global buf, size, idx, worst, total
    if not buf:
        return
    blob = comp(b"".join(buf))
    open(os.path.join(dest, f"part.{idx:04d}.{e}"), "wb").write(blob)
    worst = max(worst, len(blob)); total += len(blob); idx += 1
    buf, size = [], 0

with open(log, "rb") as fh:
    for line in fh:
        # Cut only once we are over target AND standing at a record start, so a part
        # never begins mid-record.
        if size >= target and START.match(line):
            flush()
        buf.append(line); size += len(line)
flush()
print(f"   {idx} 片 · 合计 {total:,} 字节 · 最大一片 {worst:,}", end=" ")
print("(在上限内)" if worst <= limit else f"(★超过 {limit:,},调小 RAW_TARGET 重跑)")
sys.exit(0 if worst <= limit else 3)
PYSPLIT

  local n; n="$(find "$dest" -name "part.*.$e" | wc -l)"
  cat > "$dest/MANIFEST" <<EOF
name        $name
raw_bytes   $raw_size
raw_sha256  $raw_sha
compress    $COMPRESS
raw_target  $RAW_TARGET
part_bytes  $PART_BYTES
parts       $n
restore     cat part.*.$e > $name.$e && $COMPRESS -d $name.$e && sha256sum $name
EOF
  echo "== 完成 -> $dest"
  cat "$dest/MANIFEST"
}

cmd_push() {
  local dest="${1:?usage: push <dest_dir>}"
  [ -f "$dest/MANIFEST" ] || die "no MANIFEST in $dest — run pack first"

  # Catch an ignored destination HERE, with a fix, instead of letting `git add` abort the
  # loop under `set -e` with nothing but git's generic "paths are ignored" hint.
  if git check-ignore -q "$dest" 2>/dev/null; then
    echo "!! $dest 被 .gitignore 挡住($(git check-ignore -v "$dest" | cut -f1)):" >&2
    echo "   换个不以 .log 结尾的目录名即可,分片不用重做:" >&2
    echo "     mv '$dest' '${dest%.log}'" >&2
    echo "     $0 push '${dest%.log}'" >&2
    exit 5
  fi

  local branch e
  branch="$(git rev-parse --abbrev-ref HEAD)"
  e="$(awk '$1=="compress"{print ($2=="xz")?"xz":"gz"}' "$dest/MANIFEST")"
  # MANIFEST goes first so that an interrupted run still tells the next reader what this
  # pile of parts is and how to restore it.
  local f
  for f in "$dest/MANIFEST" $(find "$dest" -name "part.*.$e" | sort); do
    # Already committed -> skip. This is what makes the whole thing resumable: after a
    # failed push, just re-run and it picks up where it stopped.
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 && continue
    git add "$f"
    git -c user.name='Sawyer117' -c user.email='wensyaustin@foxmail.com' \
        commit -q -m "logs: $(basename "$dest") $(basename "$f")"
    if ! git push -q origin "HEAD:$branch"; then
      echo >&2
      echo "!! push 失败于 $f —— 修好后重跑本命令,会从这一片继续" >&2
      exit 1
    fi
    printf '.'
  done
  echo; echo "== 全部推送完成"
}

cmd_verify() {
  local dest="${1:?usage: verify <dest_dir>}"
  [ -f "$dest/MANIFEST" ] || die "no MANIFEST in $dest"
  local name want got tool e
  name="$(awk '$1=="name"{print $2}' "$dest/MANIFEST")"
  want="$(awk '$1=="raw_sha256"{print $2}' "$dest/MANIFEST")"
  tool="$(awk '$1=="compress"{print $2}' "$dest/MANIFEST")"
  e="$([ "$tool" = "xz" ] && echo xz || echo gz)"
  # tmp is GLOBAL on purpose: the EXIT trap runs after this function returns, so a
  # `local tmp` would be out of scope there and `set -u` would abort inside the trap.
  tmp="$(mktemp -d)"; trap 'rm -rf "${tmp:-}"' EXIT
  cat $(find "$dest" -name "part.*.$e" | sort) > "$tmp/$name.$e"
  "$tool" -d "$tmp/$name.$e"
  got="$(sha256sum "$tmp/$name" | cut -d' ' -f1)"
  [ "$got" = "$want" ] || die "校验失败: want $want got $got"
  echo "== 校验通过: $name 与原始逐字节相同 ($(awk '$1=="parts"{print $2}' "$dest/MANIFEST") 片)"
}

# Deleting the parts is the one irreversible step here, so it is gated three ways:
#   1. verify must pass -- never drop the local copy on an archive we cannot rebuild;
#   2. nothing may be unpushed -- @{u}..HEAD must be empty, or GitHub does not have it yet;
#   3. --yes must be typed.
# What it removes is the parts from HEAD (git rm). They stay in history, so nothing is
# actually lost -- but a fresh clone will not see them until you check them out by sha.
cmd_clean() {
  local dest="${1:?usage: clean <dest_dir> --yes}"
  [ "${2:-}" = "--yes" ] || die "refusing without --yes (this drops the parts from HEAD)"
  cmd_verify "$dest"
  [ -z "$(git rev-list @{u}..HEAD 2>/dev/null)" ] \
    || die "还有未推送的提交 —— 先跑 push,GitHub 上没有就不能删本地"
  local sha; sha="$(git rev-parse --short HEAD)"
  git rm -r -q "$dest"
  git -c user.name='Sawyer117' -c user.email='wensyaustin@foxmail.com' \
      commit -q -m "logs: drop $(basename "$dest") from HEAD (kept in history at $sha)"
  echo "== 已从 HEAD 移除。取回:  git checkout $sha -- $dest"
  echo "   注意这只提交了删除,还没推。确认无误后:  git push origin HEAD"
}

case "${1:-}" in
  pack)   shift; cmd_pack   "$@" ;;
  push)   shift; cmd_push   "$@" ;;
  verify) shift; cmd_verify "$@" ;;
  clean)  shift; cmd_clean  "$@" ;;
  *) sed -n '2,36p' "$0"; exit 1 ;;
esac
