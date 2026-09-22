#!/usr/bin/env bash
# 可验证的 NPU 清场 —— source 进来用,别直接执行。
#
# WHY. 「pkill 之后 sleep 60 就当干净了」不是清场,是许愿。2026-09-20 实测的三种失败:
#   * 进程名单漏了 APIServer_* / DPCoordinator —— 残留进程活着,下一次起服务在
#     `torch.distributed.new_group(backend="gloo")` 报 `Failed to recv, got 0 bytes`;
#   * 进程都没了但**端口**还没放开,服务起来就撞 bind;
#   * 进程都没了、端口也放开了,**卡上还占着 61 GiB** —— SIGKILL 之后驱动收回显存要
#     12 分 40 秒,期间起服务直接报
#     `Free memory on device (22.42/61.27 GiB) ... less than desired (0.9, 55.14 GiB)`。
#     这一条最坑:报错长得像配置问题,其实是上一轮没退干净。
#
# 三层判据缺一不可,而且【显存是终极判据】—— 进程列表会漏,显存不会骗人。
#
# 用法:
#   source "$(dirname "$0")/npu_cleanup_lib.sh"
#   cleanup_verified "标签" || { echo "清不干净"; exit 1; }
#
# 可调(都给足,等长了能 Ctrl-C,等短了要人整晚盯着):
#   KILLPAT / GRACE / KILL_WAIT / PORT_WAIT / HBM_WAIT / HBM_FREE_MB / PORT
#
# ⚠ dsa_prefill_fault_ab.sh 里有一份等价实现(那个脚本已在产,不动它)。改这里的逻辑时
#   记得同步,或者把那边迁过来。

# vLLM 用 setproctitle(f"{VLLM_PROCESS_NAME_PREFIX}::{name}") 改名:VLLM::APIServer_0 /
# VLLM::DPCoordinator / VLLM::EngineCore_DP0 / VLLM::Worker,vllm-ascend 侧还有
# VLLMWorker_DP / VLLM_DP_Coordinator。都带 VLLM,但别只赌这一点。
KILLPAT="${KILLPAT:-vllm|EngineCore|APIServer|ApiServer|DPCoordinator|VLLMWorker|dspark_hs}"
GRACE="${GRACE:-60}"                 # 先 SIGTERM 让 vllm 自己收尾,等这么久再 SIGKILL
KILL_WAIT="${KILL_WAIT:-180}"        # 等进程真的退干净
PORT_WAIT="${PORT_WAIT:-300}"        # 等端口放开(TIME_WAIT)
HBM_WAIT="${HBM_WAIT:-1800}"         # 等显存回落。崩溃后驱动侧释放很慢,给足半小时
HBM_FREE_MB="${HBM_FREE_MB:-4096}"   # 单 die 已用低于这个数才算「卡是空的」(空卡通常几百)

_cl_say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
_cl_hms() { printf '%02d:%02d:%02d' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

_alive() { pgrep -i -u "$USER" -f "$KILLPAT" 2>/dev/null | grep -vx "$$" | grep -vx "$PPID"; }

# npu-smi 的 HBM-Usage 列形如 `3536 / 65536`;只取分母 >= 30000 的那组(HBM,不是旁边
# 那列小的 Memory(MB)),返回所有 die 里最大的已用值(MB)。没有 npu-smi 返回 -1。
_npu_used_mb() {
  command -v npu-smi >/dev/null 2>&1 || { echo -1; return; }
  npu-smi info 2>/dev/null \
    | grep -oE '[0-9]+ */ *[0-9]{5,}' \
    | awk -F'/' '{gsub(/ /,""); if ($2+0>=30000 && $1+0>mx) mx=$1+0} END{print mx+0}'
}

# ★ 2026-09-23:这里原来写死 `python`。未激活 conda 的 shell(以及很多发行版)只有
#   `python3`,于是 `python: command not found` 被 `2>/dev/null` 吞掉、bash 返回 127,
#   _port_free 永远为假 —— **空闲的端口被判成永久占用**。实测:一台完全空闲的 A3 上
#   determinism_sweep 等满 300 秒报「清场失败」,而 ss 里一个监听都没有,整臂作废。
#   现在:python3 优先,退回 python,再退回 ss/netstat;一个都没有就说清楚并放行。
_CL_PYBIN="${_CL_PYBIN:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)}"

_port_free() {
  local port="${PORT:-7000}"
  if [ -n "$_CL_PYBIN" ]; then
    "$_CL_PYBIN" - "$port" <<'PYEOF' 2>/dev/null
import socket, sys
s = socket.socket()
try:
    s.bind(("0.0.0.0", int(sys.argv[1]))); sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PYEOF
    return $?
  fi
  if command -v ss >/dev/null 2>&1; then
    ! ss -ltn 2>/dev/null | grep -qE "[:.]${port}[[:space:]]"
    return $?
  fi
  if command -v netstat >/dev/null 2>&1; then
    ! netstat -ltn 2>/dev/null | grep -qE "[:.]${port}[[:space:]]"
    return $?
  fi
  _cl_say "    !! 没有 python3/python/ss/netstat,端口 $port 无法验证 —— 当作空闲放行"
  return 0
}

# 返回 0 = 确认干净;返回 1 = 清不干净(调用方必须当失败处理,别硬起服务)
cleanup_verified() {
  local tag="${1:-}"
  local port="${PORT:-7000}"

  # 先 SIGTERM:vllm 自己有 graceful shutdown,让它自己退能省掉十几分钟的显存回落
  if [ -n "$(_alive)" ]; then
    pkill -TERM -i -u "$USER" -f "$KILLPAT" >/dev/null 2>&1
    local g=0
    while [ "$g" -lt "$GRACE" ] && [ -n "$(_alive)" ]; do sleep 2; g=$((g + 2)); done
    [ -n "$(_alive)" ] && _cl_say "    SIGTERM 后 $(_cl_hms $g) 还没退干净,转 SIGKILL"
  fi
  pkill -9 -i -u "$USER" -f "$KILLPAT" >/dev/null 2>&1

  local t=0
  while [ "$t" -lt "$KILL_WAIT" ] && [ -n "$(_alive)" ]; do sleep 2; t=$((t + 2)); done
  if [ "$(_alive | wc -l)" -gt 0 ]; then sleep 3; fi   # 宽限一次,别把刚退的判成残留
  local left; left=$(_alive)
  if [ -n "$left" ]; then
    local shown; shown=$(echo "$left" | while read -r pid; do
      ps -o pid=,etime=,args= -p "$pid" 2>/dev/null | cut -c1-140; done)
    if [ -n "$shown" ]; then
      _cl_say "!! 清场未完成($tag):还有进程没死:"; echo "$shown"; return 1
    fi
  fi

  local pt=0 pb=0
  while ! _port_free && [ "$pt" -lt "$PORT_WAIT" ]; do
    [ "$pt" -eq 0 ] && _cl_say "    端口 $port 还被占着,等它放开(最多 $(_cl_hms "$PORT_WAIT"))..."
    sleep 2; pt=$((pt + 2)); pb=$((pb + 2))
    if [ "$pb" -ge 60 ]; then pb=0; _cl_say "    ... 端口还没放开($(_cl_hms $pt))"; fi
  done
  if ! _port_free; then
    _cl_say "!! 清场未完成($tag):端口 $port 仍被占用($(_cl_hms $pt) 没放开)。"
    # 「判定占用」和「真的有人在监听」是两回事。没监听却判占用 = 检查器坏了,
    # 而不是端口被占 —— 把这句话直接写进日志,别让下一个人再花半小时去找幽灵进程。
    local _lsn=""
    command -v ss >/dev/null 2>&1 && _lsn="$(ss -ltnp 2>/dev/null | grep -E "[:.]${port}[[:space:]]" | head -3)"
    if [ -n "$_lsn" ]; then
      echo "$_lsn"
    else
      _cl_say "   ⚠ 但没有任何进程在监听 $port —— 不是端口被占,是这个检查本身坏了。"
      _cl_say "     端口检查用的解释器:${_CL_PYBIN:-<python3/python 都没找到>}"
    fi
    return 1
  fi

  # 泄漏的 IPC(崩溃日志里会有 "N leaked semaphore objects")。进程已确认全退才动手。
  local shm; shm=$(find /dev/shm -maxdepth 1 -user "$USER" \
                   \( -name 'psm_*' -o -name '*vllm*' -o -name 'torch_*' \) 2>/dev/null | wc -l)
  if [ "$shm" -gt 0 ]; then
    find /dev/shm -maxdepth 1 -user "$USER" \
      \( -name 'psm_*' -o -name '*vllm*' -o -name 'torch_*' \) -delete 2>/dev/null
  fi

  # ★ 终极判据
  local used; used=$(_npu_used_mb)
  if [ "$used" -ge 0 ] 2>/dev/null; then
    local ht=0 hb=0
    while [ "$used" -gt "$HBM_FREE_MB" ] && [ "$ht" -lt "$HBM_WAIT" ]; do
      [ "$ht" -eq 0 ] && _cl_say "    卡上还占着 ${used} MiB,等显存回落(最多 $(_cl_hms "$HBM_WAIT"))..."
      sleep 10; ht=$((ht + 10)); used=$(_npu_used_mb); hb=$((hb + 10))
      if [ "$hb" -ge 60 ]; then hb=0; _cl_say "    ... 显存还没清空($(_cl_hms $ht))"; fi
    done
    if [ "$used" -gt "$HBM_FREE_MB" ]; then
      _cl_say "!! 清场未完成($tag):卡上仍有 ${used} MiB 被占(阈值 ${HBM_FREE_MB} MiB,等了 $(_cl_hms $ht))。"
      _cl_say "   占用纹丝不动 = 有进程没杀掉;还在往下走 = 这机器放得慢,加大 HBM_WAIT。"
      npu-smi info 2>/dev/null | grep -iE 'vllm|python|process id' | head -20
      return 1
    fi
    [ "$ht" -gt 0 ] && _cl_say "    显存在 $(_cl_hms $ht) 后回落到 ${used} MiB"
    _cl_say "    清场已核实:无残留进程,端口 $port 可绑定,卡上占用 ${used} MiB"
  else
    _cl_say "    清场已核实:无残留进程,端口 $port 可绑定(没有 npu-smi,跳过显存检查)"
  fi
  return 0
}
