#!/usr/bin/env bash
# 把一条训练 run 的【原始全量日志】归档进 article 仓 —— 一条命令,可重复跑。
#
# WHY
# ---
# 这条链路上有个硬约束:**网关对单次 HTTP 请求体的上限约 100 KB**。`git push` 把整个
# pack 作为一个 POST 发出去,所以**上限落在 push 上,不落在文件上** —— 切文件不够,
# 每一次 push 也必须小于它。`archive_log_push.sh` 就是为这个写的(一片一提交一推、
# 可断点续),本脚本只是把「定位 → 脱敏 → 切片 → 推 → 校验」串成一条命令。
#
# 为什么保全量而不是采样:采样(`split_train_log.py --every N`)会丢掉 spike 分析
# ——49 个 spike 全是单步事件,1/100 抽样抽不到;而且分析器的滚动窗口按行数算,
# 采样后和基线那份全分辨率归档不可比。原始日志才是唯一真相,csv.gz 是衍生物,
# 任何时候都能从它重新生成。
#
# ★ 对【还在跑】的 run 基本是增量的:pack 按记录边界切分是确定性的,日志长大后重跑,
#   **除了上一次那个未填满的尾片**,前面的分片字节完全相同,git 认得出、push 自动跳过。
#   实测(合成用例,12 片 → 16 片):11/12 片一模一样,只重推 1 个旧尾片 + 4 个新片。
#
# USAGE
#   bash archive_run_to_article.sh                      # 归档 $RUN 里最新的 *.log
#   bash archive_run_to_article.sh <logfile>
#   NAME=blk15_bal bash archive_run_to_article.sh <logfile>    # 自定归档目录名
#
# ENV
#   RUN      训练 run 目录          默认 /home/a00652497/dspark_austin/run
#   ART      article 仓路径         默认 <RUN 的上级>/dsv4-dspark-article(没有就 clone)
#   NAME     归档子目录名           默认从日志名推断
#   REDACT   1=先脱敏(默认)       0=直接打包(article 仓是私有的,可以关)
#   PART_BYTES / RAW_TARGET        透传给 archive_log_push.sh(默认 90000 / 1800000)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${RUN:-/home/a00652497/dspark_austin/run}"
ART="${ART:-$(dirname "$RUN")/dsv4-dspark-article}"
REDACT="${REDACT:-1}"
ART_URL="${ART_URL:-https://github.com/Sawyer117/dsv4-dspark-article.git}"

say() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "!! $*" >&2; exit 1; }

LOG="${1:-$(ls -1t "$RUN"/*.log 2>/dev/null | head -1)}"
[ -n "$LOG" ] && [ -f "$LOG" ] || die "找不到日志(给个路径,或设 RUN=)"
NAME="${NAME:-$(basename "$LOG" .log)}"
DEST="experiments/logs/${NAME}_raw"

echo "================================================================================"
echo "  日志    $LOG   ($(du -h "$LOG" | cut -f1))"
echo "  仓库    $ART"
echo "  归档到  $DEST"
echo "  脱敏    $([ "$REDACT" = 1 ] && echo 是 || echo '否(REDACT=0)')"
echo "  每片    ${PART_BYTES:-90000} B 压缩后(网关单请求上限 ~100 KB)"
echo "================================================================================"

# ── 1. article 仓:有就用,没有才 clone ──────────────────────────────────────
if [ -d "$ART/.git" ]; then
  say "article 仓已存在,拉最新"
  git -C "$ART" pull -q --ff-only origin main || say "⚠ pull 没成功(本地有改动?),继续"
else
  say "clone article 仓 → $ART"
  git clone -q "$ART_URL" "$ART" || die "clone 失败(私有仓,检查凭据)"
fi

# ── 2. 脱敏 ────────────────────────────────────────────────────────────────
SRC="$LOG"
if [ "$REDACT" = "1" ]; then
  SRC="$HOME/${NAME}.redacted.log"
  if [ -f "$SRC" ] && [ "$SRC" -nt "$LOG" ]; then
    say "脱敏文件已是最新,跳过"
  else
    say "脱敏 → $SRC(原文件只读,不动)"
    python "$SCRIPT_DIR/redact_log.py" "$LOG" "$SRC" || die "脱敏失败"
  fi
fi

# ── 3. 切片打包(确定性:重跑只会多出尾部的新分片)──────────────────────────
say "切片打包 → $ART/$DEST"
mkdir -p "$ART/$DEST"
ALLOW_UNREDACTED="${ALLOW_UNREDACTED:-$([ "$REDACT" = 1 ] && echo 0 || echo 1)}" \
  bash "$SCRIPT_DIR/archive_log_push.sh" pack "$SRC" "$ART/$DEST" || die "pack 失败"
say "分片 $(find "$ART/$DEST" -name 'part.*' | wc -l) 个,合计 $(du -sh "$ART/$DEST" | cut -f1)"

# ── 4. 一片一提交一推(可断点续:失败后重跑本脚本即可)────────────────────────
say "开始推送(一片一次请求;中断后重跑本脚本会从断点继续)"
( cd "$ART" && bash "$SCRIPT_DIR/archive_log_push.sh" push "$DEST" ) || {
  echo "!! 推送中断。修好网络/凭据后【重跑本脚本】,会从断掉的那一片继续。" >&2
  exit 1
}

# ── 5. 校验:重组后和源文件比 sha256 ────────────────────────────────────────
say "校验(重组 → sha256 对拍)"
# 把源日志也传进去:sha 对不上时 verify 会再判一次「重组结果是不是源日志的前缀」。
# 退出码 4 = 数据完好、只是 MANIFEST 的 sha 算早了(旧版 pack 有过这个两趟读的竞态);
# 对一条还在跑的 run 这不是损坏,重跑本脚本就会让 MANIFEST 自洽。
( cd "$ART" && bash "$SCRIPT_DIR/archive_log_push.sh" verify "$DEST" "$SRC" )
case $? in
  0) ;;
  4) say "⚠ 归档数据完好,但 MANIFEST 的 sha 是旧版 pack 算早的 —— 重跑本脚本即可自洽" ;;
  *) die "校验不通过 —— 别信这份归档" ;;
esac

echo
echo "================================================================================"
say "完成。$ART/$DEST"
echo "  分析器可以直接吃原始日志;要 csv.gz 的话本地生成即可,不必入库:"
echo "    python $SCRIPT_DIR/split_train_log.py $SRC --out ~/${NAME}_split --gzip"
echo "  run 还在跑的话,过些天重跑本脚本 —— 只重推上次那个未填满的尾片 + 新增的片。"
echo "================================================================================"
