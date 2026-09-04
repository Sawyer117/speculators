#!/usr/bin/env bash
# One-shot, READ-ONLY diagnostic for an Ascend NPU collective-communication hang.
#
# WHY. A training run that completes forward and backward -- hundreds of collectives --
# and then stalls inside one HCCL op with `EI0002 ... Stuck Occurred` is NOT diagnosable
# from the Python traceback: the traceback names whichever collective was issued last,
# not the one that broke. The evidence lives in npu-smi, the driver ring, and the Ascend
# plog, and it has to be collected BEFORE anyone resets a card, because a reset destroys
# exactly the state that says what happened.
#
# Nothing here writes, kills, or resets anything. Safe on a shared box.
#
#   bash npu_hang_diag.sh 2>&1 | tee /tmp/npu_diag.out
set -o pipefail

hr() { printf '\n════════ %s ════════\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
NPU_IDS="${NPU_IDS:-0 1 2 3 4 5 6 7}"
SINCE="${SINCE:-2 hours}"

hr "0. 上下文"
date
echo "host    : $(hostname)  ($(hostname -I 2>/dev/null | awk '{print $1}'))"
echo "user    : $(whoami)"
echo "uptime  : $(uptime -p 2>/dev/null || uptime)"
if [ -d .git ]; then
  echo "repo    : $(git log --oneline -1 2>/dev/null)"
  echo "branch  : $(git branch --show-current 2>/dev/null || echo '(detached HEAD)')"
fi
echo "CANN    : ${ASCEND_HOME_PATH:-<unset>}"

# Matching on the command line alone counts THIS script, whose own text contains the
# pattern -- a false positive that reads as "there are leftovers" when there are none.
# Exclude our own pid and the diag name.
leftovers() {
  ps -eo pid,user,etime,cmd 2>/dev/null \
    | awk -v me="$$" '$1 != me && /scripts\/train\.py|torchrun/ && !/npu_hang_diag|awk/'
}
hr "1. 残留训练进程"
leftovers | head -20
echo "  -> 训练相关进程数: $(leftovers | wc -l)  (期望 0)"

hr "2. npu-smi info 总览"
have npu-smi && npu-smi info 2>&1 | head -40 || echo "  npu-smi 不可用"

hr "3. 每张卡:占用进程 + 健康 + 错误码"
for i in $NPU_IDS; do
  echo "--- device $i ---"
  have npu-smi || break
  npu-smi info -t proc-mem -i "$i" 2>&1 | grep -viE "^$|^\s*\|=*\|$" | head -6
  npu-smi info -t health   -i "$i" 2>&1 | grep -iE "health|error|status" | head -4
done

hr "4. 别人的 NPU 任务(共享机!)"
# Only real workloads: python/vllm with >200 MB RSS. A bare interpreter or a shell
# wrapper is not competing for a card, and listing them buries the one that is.
ps -eo user,pid,etime,rss,cmd --sort=-rss 2>/dev/null \
  | awk 'NR==1 || ($4 > 200000 && /python|torchrun|vllm/ && !/npu_hang_diag|awk/)' | head -15
echo "  -> 只列了 RSS>200MB 的;有别人的长时间任务就可能在抢卡或抢通信"

hr "5. IPC / 共享内存残留"
echo "信号量集合数: $(ipcs -s 2>/dev/null | grep -c '^0x')"
echo "共享内存段数: $(ipcs -m 2>/dev/null | grep -c '^0x')"
ipcs -m 2>/dev/null | awk 'NR>3 && $6==0 && $5>1000000 {n++} END{print "  已分离(nattch=0)且 >1MB 的段: " (n+0)}'
echo "/dev/shm 条目: $(ls /dev/shm 2>/dev/null | wc -l)"
ls -lt /dev/shm 2>/dev/null | head -6

hr "6. HCCL / rendezvous 端口占用"
if have ss; then ss -lntp 2>/dev/null | grep -E ":(2[0-9]{4}|6[0-9]{4})" | head -12
else netstat -lntp 2>/dev/null | grep -E ":(2[0-9]{4}|6[0-9]{4})" | head -12; fi
echo "  (torchrun 用 29500 附近,HCCL 用高位端口;残留监听 = 上次没退干净)"

hr "7. 驱动 / 内核报错"
dmesg -T 2>/dev/null | grep -iE "davinci|hisi|npu|drv_|hccs|pcie" | tail -25 \
  || echo "  dmesg 不可读(需要权限),跳过"

hr "8. Ascend plog —— ★ 最接近真相的一层"
LOGDIR="${ASCEND_PROCESS_LOG_PATH:-$HOME/ascend/log}"
echo "plog 根目录: $LOGDIR"
if [ -d "$LOGDIR" ]; then
  echo "-- 最近 $SINCE 内改动过的日志文件(前 5)--"
  find "$LOGDIR" -name "*.log" -newermt "-$SINCE" 2>/dev/null | head -5
  echo
  echo "-- 关键字命中统计 --"
  for kw in "error cqe" "Stuck Occurred" "EI0002" "EI0006" "link down" "retry"; do
    n=$(find "$LOGDIR" -name "*.log" -newermt "-$SINCE" 2>/dev/null \
        | xargs grep -ilF "$kw" 2>/dev/null | wc -l)
    printf "  %-16s 命中文件数: %s\n" "$kw" "$n"
  done
  echo
  echo "-- 'error cqe' 上下文(若有,这是链路层丢包的硬证据)--"
  find "$LOGDIR" -name "*.log" -newermt "-$SINCE" 2>/dev/null \
    | xargs grep -ihF "error cqe" 2>/dev/null | tail -15
  echo
  echo "-- 'Stuck' 上下文 --"
  find "$LOGDIR" -name "*.log" -newermt "-$SINCE" 2>/dev/null \
    | xargs grep -ihE "Stuck Occurred|stuck notify" 2>/dev/null | tail -10
else
  echo "  目录不存在 —— 换个位置找:"
  ls -d /root/ascend/log /var/log/npu/slog "$HOME"/ascend* 2>/dev/null | head
fi

hr "9. HS 服务是否在(训练依赖)"
for ep in "${ENDPOINT:-http://80.5.5.115:7000/v1}" "http://80.5.5.116:7000/v1"; do
  if curl -sf --noproxy '*' --max-time 5 "$ep/models" >/dev/null 2>&1; then
    echo "  OK   $ep"
  else
    echo "  DOWN $ep"
  fi
done

hr "10. 小结(照抄给排查的人)"
echo "训练残留进程 : $(leftovers | wc -l)"
have npu-smi && echo "卡上有进程   : $(npu-smi info 2>/dev/null | grep -cE '^\| *[0-9]+ +[0-9]+ ')"
echo "error cqe    : $(find "${LOGDIR:-/nonexistent}" -name '*.log' -newermt "-$SINCE" 2>/dev/null | xargs grep -ilF 'error cqe' 2>/dev/null | wc -l) 个文件命中"
echo
echo "读法:"
echo "  error cqe 有命中          -> 卡间链路丢包,硬件/链路层,复位或报修"
echo "  残留进程或卡上有进程       -> 先清干净再谈其他"
echo "  以上都干净                -> 不是残留也不是链路,回到软件侧继续切"
