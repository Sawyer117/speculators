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

# The manifest is MANIFEST.txt, not MANIFEST: the repo's .gitignore carries a bare
# `MANIFEST` (the setuptools-generated file, straight out of the Python template), which
# matches at any depth -- so a plain MANIFEST was unaddable. Older packs still have one;
# prefer the new name and fall back, so an existing archive does not need re-packing.
mf() { [ -f "$1/MANIFEST.txt" ] && echo "$1/MANIFEST.txt" || echo "$1/MANIFEST"; }

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

  # ★ 打包一个【还在被追加写】的日志时,绝不能分两趟读。
  #   原来是先 `sha256sum $log`(558 MB 要几秒)、再让 python 重读一遍切片 —— 两趟之间
  #   训练又写进去好几步(每 3.2 s 约 26 行),于是分片包含的内容比被哈希的那段多,
  #   verify 必然失败。2026-09-22 实测:325 片全部推完,最后 sha 对不上,而分片本身是好的。
  #   修法:先 stat 取一个快照大小当硬边界,然后**在切片的同一趟里**对真正写进分片的
  #   字节算 sha —— 哈希的和打包的按定义就是同一批。
  local raw_sha raw_size snap
  snap="$(stat -c%s "$log")"

  echo "== 切片 + 压缩 ($COMPRESS, 每片约 $RAW_TARGET 未压缩字节, 只在记录边界落刀)"
  rm -f "$dest"/part.* "$dest/.packed"   # 陈旧的 .packed 会让下面读到上一次的 sha
  COMPRESS="$COMPRESS" RAW_TARGET="$RAW_TARGET" PART_BYTES="$PART_BYTES" \
    python3 - "$log" "$dest" "$e" "$snap" "$dest/.packed" <<'PYSPLIT'
import gzip, hashlib, lzma, os, re, sys
log, dest, e = sys.argv[1], sys.argv[2], sys.argv[3]
snap, sidecar = int(sys.argv[4]), sys.argv[5]                # 快照边界:只打包这么多字节,不追着长大的文件跑
target = int(os.environ["RAW_TARGET"])
limit = int(os.environ["PART_BYTES"])
h = hashlib.sha256()                   # ★ 只对真正写进分片的字节算,和打包同一趟
consumed = 0
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
        # 跨过快照边界的那一行整行不要 —— 保证切口落在行边界上,而且哈希与分片一致。
        if consumed + len(line) > snap:
            break
        # Cut only once we are over target AND standing at a record start, so a part
        # never begins mid-record.
        if size >= target and START.match(line):
            flush()
        buf.append(line); size += len(line)
        h.update(line); consumed += len(line)
flush()
print(f"   {idx} 片 · 合计 {total:,} 字节 · 最大一片 {worst:,}", end=" ")
print("(在上限内)" if worst <= limit else f"(★超过 {limit:,},调小 RAW_TARGET 重跑)")
open(sidecar, "w").write(f"{h.hexdigest()} {consumed}\n")
sys.exit(0 if worst <= limit else 3)
PYSPLIT
  # 切片脚本把「真正打进分片的」sha 和字节数写在 $dest/.packed 里(边信道,不和进度
  # 输出抢 stdout,nohup 下也不依赖 tty)。
  [ -f "$dest/.packed" ] || die "拿不到打包后的 sha —— 切片脚本没写 .packed"
  raw_sha="$(awk '{print $1}' "$dest/.packed")"
  raw_size="$(awk '{print $2}' "$dest/.packed")"
  [ -n "$raw_sha" ] && [ -n "$raw_size" ] || die ".packed 内容不对:$(cat "$dest/.packed")"
  rm -f "$dest/.packed"                  # 边信道用完即弃,别留在仓库工作区里
  [ "$raw_size" = "$snap" ] || echo "   (快照 $snap B,按行边界收敛到 $raw_size B —— 末尾半行已舍去)"

  local n; n="$(find "$dest" -name "part.*.$e" | wc -l)"
  cat > "$dest/MANIFEST.txt" <<EOF
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
  cat "$dest/MANIFEST.txt"
}

cmd_push() {
  local dest="${1:?usage: push <dest_dir>}"
  local mfp; mfp="$(mf "$dest")"
  [ -f "$mfp" ] || die "no MANIFEST.txt in $dest — run pack first"

  local branch e
  branch="$(git rev-parse --abbrev-ref HEAD)"
  e="$(awk '$1=="compress"{print ($2=="xz")?"xz":"gz"}' "$mfp")"

  # Catch anything ignored HERE, with a fix, instead of letting `git add` abort the loop
  # under `set -e` with nothing but git's generic "paths are ignored" hint. Check EVERY
  # file about to be added, not just the directory: checking only the directory passed,
  # then `git add` died on MANIFEST, which .gitignore matches by its own bare name.
  local blocked
  blocked="$(git check-ignore "$mfp" $(find "$dest" -name "part.*.$e" | sort) 2>/dev/null || true)"
  if [ -n "$blocked" ]; then
    echo "!! 这些路径被 .gitignore 挡住,git add 会失败:" >&2
    git check-ignore -v $blocked >&2
    case "$blocked" in
      *MANIFEST) echo "   修法:  mv '$dest/MANIFEST' '$dest/MANIFEST.txt'  然后重跑本命令" >&2 ;;
      *) echo "   修法:换一个不被忽略的目录名,分片不用重做:" >&2
         echo "     mv '$dest' '${dest%.log}' && $0 push '${dest%.log}'" >&2 ;;
    esac
    exit 5
  fi

  # A previous run can leave commits made but not pushed (commit succeeds, push fails on
  # auth). Drain them BEFORE adding more, or the next push carries several parts in one
  # request -- which is the very thing the per-part split exists to avoid.
  if [ -n "$(git rev-list @{u}..HEAD 2>/dev/null)" ]; then
    echo "== 先把上次已提交但未推送的 $(git rev-list --count @{u}..HEAD) 个提交推掉"
    git push -q origin "HEAD:$branch" || die "push 失败 —— 认证/网络修好后重跑本命令"
  fi

  # MANIFEST goes first so that an interrupted run still tells the next reader what this
  # pile of parts is and how to restore it.
  local f
  for f in "$mfp" $(find "$dest" -name "part.*.$e" | sort); do
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

# verify <dest_dir> [原始日志]
# 给了原始日志、而 sha 又对不上时,会再判一次「重组结果是不是原始日志的前缀」——
# 对一条【还在跑】的 run,这才是真正要问的问题:数据完好但源文件之后又长了(正常),
# 还是分片本身坏了(严重)。旧版本的 pack 有过一个两趟读的竞态会造成前者。
cmd_verify() {
  local dest="${1:?usage: verify <dest_dir> [原始日志]}"
  local src="${2:-}"
  local mfp; mfp="$(mf "$dest")"
  [ -f "$mfp" ] || die "no MANIFEST.txt in $dest"
  local name want got tool e
  name="$(awk '$1=="name"{print $2}' "$mfp")"
  want="$(awk '$1=="raw_sha256"{print $2}' "$mfp")"
  tool="$(awk '$1=="compress"{print $2}' "$mfp")"
  e="$([ "$tool" = "xz" ] && echo xz || echo gz)"
  # tmp is GLOBAL on purpose: the EXIT trap runs after this function returns, so a
  # `local tmp` would be out of scope there and `set -u` would abort inside the trap.
  tmp="$(mktemp -d)"; trap 'rm -rf "${tmp:-}"' EXIT
  cat $(find "$dest" -name "part.*.$e" | sort) > "$tmp/$name.$e"
  "$tool" -d "$tmp/$name.$e"
  got="$(sha256sum "$tmp/$name" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "!! sha 对不上: want $want" >&2
    echo "               got  $got" >&2
    if [ -n "$src" ] && [ -f "$src" ]; then
      local n; n="$(stat -c%s "$tmp/$name")"
      if [ "$(stat -c%s "$src")" -ge "$n" ] && cmp -s -n "$n" "$tmp/$name" "$src"; then
        echo "   但重组结果是 $src 的【前缀】($n B,源文件现在 $(stat -c%s "$src") B)。" >&2
        echo "   ⟹ 分片数据是好的,只是 MANIFEST 的 sha 算早了 / 源文件之后又长了。" >&2
        echo "   重跑 pack 即可让 MANIFEST 自洽(旧分片字节相同,push 会自动跳过)。" >&2
        return 4
      fi
      echo "   而且它【不是】$src 的前缀 —— 分片真的坏了,别信这份归档。" >&2
    else
      echo "   传第二个参数(原始日志路径)可以再判一次它是不是前缀:" >&2
      echo "     bash $0 verify $dest <原始日志>" >&2
    fi
    return 1
  fi
  echo "== 校验通过: $name 与原始逐字节相同 ($(awk '$1=="parts"{print $2}' "$mfp") 片)"
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
