#!/usr/bin/env bash
# DSV4-DSpark spec-decode eval CLIENT — run AFTER the serve is up (serve_dsv4_bf16_dualnode.sh head).
# Does NOT start a serve. Steps: wait-for-ready -> smoke test (数数) -> spec-decode counter check ->
# gsm8k accept_len benchmark (Evaluator.py, greedy). One command, git-pullable.
#
#   bash run_dspark_eval.sh                 # defaults: port 7000, model dsv4, gsm8k, concurrency 16
#   PORT=7000 DATASET=gsm8k CONCURRENCY=16 bash run_dspark_eval.sh
#
# Bar to beat = released DSV4 draft accept_len 3.94 @ num_spec=5 (vllm-ascend PR #11196).
set -o pipefail
PORT="${PORT:-7000}"
MODEL="${MODEL:-dsv4}"
DATASET="${DATASET:-gsm8k}"          # only gsm8k by default — /metrics counters reset mid-run, so a
                                     # single dataset is the trustworthy accept_len (see memory).
CONCURRENCY="${CONCURRENCY:-16}"     # bf16 dual-node: keep <=64 client (KV overflow -> garbage above that).
NUM_PROMPTS="${NUM_PROMPTS:-0}"      # 0 = full dataset (1319 for gsm8k, ~2h @ conc16). 200 is a stable
                                     # accept_len in ~10min. accept_len is a per-token average — a few
                                     # hundred prompts is plenty; full run only tightens the estimate.
MAX_NEW="${MAX_NEW:-2048}"
# LOCAL tokenizer dir — --model is the served-name `dsv4` (for the API), NOT a HF repo, so the tokenizer
# must load from a local path (the bf16 target has tokenizer.json). Override via TOKENIZER=<dir>.
# Auto-detect the box's ckpt root: A3-176 = /home/canada_group_folder, A2 = /share, A3-nfs = /mnt/nfs.
if [ -z "${TOKENIZER:-}" ]; then
  for _d in /home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 \
            /share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 \
            /mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16; do
    [ -d "$_d" ] && TOKENIZER="$_d" && break
  done
fi
TOKENIZER="${TOKENIZER:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
BASE="http://localhost:$PORT"
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"   # corp proxy hijacks localhost
# box HF access is flaky → force datasets/hub to the LOCAL cache (~/.cache/huggingface). Set OFFLINE=0 to
# allow network (e.g. to first-time download a dataset). The datasets must already be cached under OFFLINE=1.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-${OFFLINE:-1}}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-${OFFLINE:-1}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# DSV4 tokenizer ships NO chat_template → apply this jinja (ported for training data prep). Override via CHAT_TEMPLATE=.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/dsv4_chat_template.jinja}"

echo ">>> [1/4] waiting for serve on :$PORT (up to 600s) ..."
for _ in $(seq 1 120); do
  curl -sf --noproxy '*' "$BASE/v1/models" >/dev/null 2>&1 && { echo "    READY"; break; }
  sleep 5
done
curl -sf --noproxy '*' "$BASE/v1/models" >/dev/null 2>&1 || { echo "!! serve not ready on :$PORT — is the head up? tail ~/dsv4_bf16_head.log"; exit 1; }

echo ">>> [2/4] smoke test (数数 — DSpark crashes HERE on the first draft if misconfigured) ..."
OUT=$(curl -s --noproxy '*' "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"从1数到40，用空格分隔\"}],\"max_tokens\":128,\"temperature\":0}")
echo "$OUT" | python3 -c "import sys,json;d=json.load(sys.stdin);print('   ->', d['choices'][0]['message']['content'][:160])" 2>/dev/null \
  || { echo "!! smoke request FAILED (draft likely crashed — dim mismatch / dead worker). Raw:"; echo "$OUT" | head -c 600; echo; echo "   tail ~/dsv4_bf16_head.log for the traceback."; exit 2; }

echo ">>> [3/4] spec-decode counters (must be NON-zero to prove drafting is live) ..."
curl -s --noproxy '*' "$BASE/metrics" | grep -E "spec_decode_num_(drafts|accepted_tokens)_total" \
  || echo "    (!! no spec_decode counters — serve fell back to AR? check speculative_config in ~/dsv4_bf16_head.log)"

echo ">>> [4/4] $DATASET accept_len (Evaluator, greedy temp0/top-p1/top-k1) — bar = released 3.94 ..."
NP_ARG=(); [ "$NUM_PROMPTS" -gt 0 ] 2>/dev/null && NP_ARG=(--num-prompts "$NUM_PROMPTS")
exec python "$SCRIPT_DIR/Evaluator.py" \
  --base-url "$BASE" --model "$MODEL" --tokenizer "$TOKENIZER" --chat-template "$CHAT_TEMPLATE" \
  --dataset "$DATASET" "${NP_ARG[@]}" \
  --concurrency "$CONCURRENCY" --warmup-steps 10 --max-new-tokens "$MAX_NEW" \
  --temperature 0 --top-p 1 --top-k 1
