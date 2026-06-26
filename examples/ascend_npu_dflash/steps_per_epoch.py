#!/usr/bin/env python3
"""Compute steps-per-epoch (= len(train_loader)) for a prepared Arrow dataset,
WITHOUT training and WITHOUT touching the NPU.

The trainer never logs this number — it only appears in the tqdm.rich bar, which
does not render under nohup (non-TTY). But it is fully deterministic: the trainer
builds `MultipackDistributedBatchSamplerV2(batch_max_length=total_seq_len,
lengths=dataset.approx_lengths, num_replicas=world_size, rank=...)` and
`len(train_loader) == len(batch_sampler)`. `approx_lengths` is exactly the Arrow
`seq_len` column (data.py: ArrowDataset._compute_approx_lengths). So we reuse the
SAME sampler on the SAME lengths to get the SAME count — CPU only, a few seconds.

Feed the result to train_timing.py for an exact ETA:
    python train_timing.py logs/ --total-steps <total>

Usage:
    python steps_per_epoch.py \
        --data-path /share/.../open_perfectblend.qwen3-4b-rollout.qwen3.seq3072 \
        --world-size 6 --total-seq-len 3072 --epochs 1
"""
# SPDX-License-Identifier: Apache-2.0
import argparse

import numpy as np
from datasets import load_from_disk

from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2


def main():
    ap = argparse.ArgumentParser(description="steps/epoch for a prepared Arrow dataset.")
    ap.add_argument("--data-path", required=True, help="prepared Arrow dir (has seq_len)")
    ap.add_argument("--world-size", type=int, required=True,
                    help="FSDP world size = NPROC (trainer cards). 2+6 baseline → 6")
    ap.add_argument("--total-seq-len", type=int, default=3072,
                    help="per-rank token budget = --total-seq-len (must match training)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="sampler seed (trainer default 0)")
    args = ap.parse_args()

    d = load_from_disk(args.data_path)
    # exactly what ArrowDataset.approx_lengths returns (data.py:_compute_approx_lengths)
    lengths = list(d.with_format(None)["seq_len"])

    sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=args.total_seq_len,
        lengths=lengths,
        num_replicas=args.world_size,
        rank=0,
        seed=args.seed,
    )
    steps_per_epoch = len(sampler)            # len(train_loader) at epoch 0
    total = steps_per_epoch * args.epochs     # = trainer's scheduler_total_steps

    sl = np.array(lengths)
    print("=" * 60)
    print(f"data             : {args.data_path}")
    print(f"samples          : {len(sl):,}  (seq_len mean {sl.mean():.0f}, max {int(sl.max())})")
    print(f"batch_max_length : {args.total_seq_len} tokens/rank")
    print(f"world_size(NPROC): {args.world_size}")
    print(f"steps/epoch      : {steps_per_epoch:,}")
    print(f"epochs           : {args.epochs}")
    print(f"TOTAL steps      : {total:,}")
    print("=" * 60)
    print("→ 喂给 train_timing 拿精确 ETA:")
    print(f"   python examples/ascend_npu_dflash/train_timing.py logs/ --total-steps {total}")


if __name__ == "__main__":
    main()
