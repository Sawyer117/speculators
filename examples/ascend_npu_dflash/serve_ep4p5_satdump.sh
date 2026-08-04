#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────────────────
# Serve the ep4p5 DSpark draft on A3 (176, TP8×DP2) in EAGER mode with BOTH dumps armed, for
# the train↔serve forward bisection. After startup send ONE batch=1 temp=0 request; the serve
# writes  $HOME/dspark_parity/serve_block_0.pt  (parity: aux+base_logits)  AND
#         $HOME/dspark_sat/serve_sat.pt         (saturated: per-layer + 8 sub-stages).
#
# Requires the vllm-ascend `dspark-parity` build (has the DSPARK_PARITY_DUMP + DSPARK_SATDUMP
# hooks). On the serve box:  cd <vllm-ascend> && git pull   # dspark-parity @ >= 72e4efbc
#
# FULL FLOW (kept here for traceability):
#   1) this script (176) → serve_block_0.pt + serve_sat.pt
#   2) send request:
#        curl --noproxy '*' http://localhost:7000/v1/completions -H 'Content-Type: application/json' \
#             -d '{"model":"dsv4","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
#   3) scp BOTH (same run) to 109:
#        scp $HOME/dspark_parity/serve_block_0.pt  <109>:~/dspark_parity/
#        scp $HOME/dspark_sat/serve_sat.pt         <109>:~/dspark_sat/
#   4) on 109:
#        DSPARK_SATDUMP=1 DSPARK_SATDUMP_DIR=~/dspark_sat python examples/ascend_npu_dflash/dsv4_dspark_forward_parity_v2.py \
#          --dumps ~/dspark_parity --ckpt /home/a00652497/dspark_austin/run/ckpt_faithful_ep_20260729_092941/1 \
#          --verifier /share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 --device npu --dtype bfloat16
#        python examples/ascend_npu_dflash/dsv4_dspark_satdump_compare.py \
#          --serve ~/dspark_sat/serve_sat.pt --train ~/dspark_sat/train_sat.pt
# ─────────────────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VERIFIER=/home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16
DRAFT=/home/canada_group_folder/ckpt/dsv4_dspark_ep4p5_vllm-77w
PORT=7000

export no_proxy=localhost,127.0.0.1,::1                       # else curl localhost is proxy-hijacked
export ASCEND_A3_ENABLE=1 VLLM_ASCEND_ENABLE_FUSED_MC2=1 HCCL_BUFFSIZE=1024
export EAGER=1                                                 # disable drafter ACLGraph → Python dump fires
export DSPARK_PARITY_DUMP=1 DSPARK_PARITY_DIR="$HOME/dspark_parity"
export DSPARK_SATDUMP=1     DSPARK_SATDUMP_DIR="$HOME/dspark_sat"

exec vllm serve "$VERIFIER" \
  --served-model-name dsv4 \
  --port "$PORT" \
  --data-parallel-size 2 --data-parallel-size-local 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --block-size 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --enforce-eager \
  --additional-config '{"enable_cpu_binding":true}' \
  --speculative-config "{\"method\":\"mtp\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":5,\"enforce_eager\":true}"
