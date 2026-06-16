#!/usr/bin/env bash
set -euo pipefail

ENDPOINT=${ENDPOINT:-http://80.5.5.108:30000/v1/chat/completions}
TARGET=${TARGET:-/home/n84449292/m84379596/Huggingface/DeepSeek-V4-Flash-bf16}
NO_PROXY_HOSTS=${NO_PROXY_HOSTS:-80.5.5.108,80.5.5.109,localhost,127.0.0.1}

for n in 1 2 4 8 16; do
  echo
  echo "===== max_tokens=$n ====="
  curl --noproxy "$NO_PROXY_HOSTS" \
    -s "$ENDPOINT" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"$TARGET\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer only the number.\"}],
      \"max_tokens\": $n,
      \"temperature\": 0
    }"
done
echo
