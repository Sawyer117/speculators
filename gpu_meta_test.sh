#!/usr/bin/env bash
# GPU-side validation helper for PR #776 (--init-on-meta meta-device guard + tests).
# NOT part of the PR — lives only on the scratch branch pr/init-on-meta-review.
#
# Usage (on the GPU box):
#   git clone --branch pr/init-on-meta-review --single-branch \
#     https://github.com/Sawyer117/speculators speculator-gpu
#   cd speculator-gpu
#   bash gpu_meta_test.sh
set -euo pipefail

ENV_NAME=speculator-gpu
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> [1/3] 新建 conda 环境 ${ENV_NAME} (python 3.11)"
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
conda create -n "${ENV_NAME}" python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> [2/3] 安装 speculators (editable) + pytest  (repo: ${REPO_DIR})"
cd "${REPO_DIR}"
pip install -U pip
pip install -e .
pip install pytest

echo "==> [3/3] 跑 meta-init 测试"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pytest tests/unit/train/test_build_on_meta.py -v

echo
echo "全 PASSED 就把上面输出发回。可选:确认基类 model.py guard 没弄坏别的 ->"
echo "  pytest tests/unit/train/ tests/unit/models/ -q"
