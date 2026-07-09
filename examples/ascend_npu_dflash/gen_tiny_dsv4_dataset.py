#!/usr/bin/env python3
"""Generate a TINY synthetic DSV4 dataset to exercise scripts/train.py end-to-end.

Produces exactly what ArrowDataset(on_missing="skip") consumes — so the Step-1
trainer-machinery gate (build via from_training_args + FSDP2 + dataloader +
train loop + checkpoint) runs on a single card WITHOUT the heavy HS-producer
serve. The hidden states are RANDOM (this validates the machinery, not learning).

Layout (matches src/speculators/train/data.py::ArrowDataset._get_raw_data):
  <out>/arrow/                 HF dataset (save_to_disk) with columns:
                                 input_ids [seq], loss_mask [seq], seq_len (int)
  <out>/hs/hs_<i>.safetensors  { "hidden_states": [seq, num_layers, H] bf16,
                                 "token_ids": [seq] int64 }  (token_ids == input_ids)
  num_layers = len(target_layers)+1 (aux layers [40,41,42] + verifier-last).

Run on the box (needs datasets + safetensors)::

  python gen_tiny_dsv4_dataset.py --out /home/a00652497/dspark_2026/runs/tiny_ds
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir (creates arrow/ + hs/)")
    ap.add_argument("--n", type=int, default=32, help="number of samples")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=129280)
    ap.add_argument("--num-aux", type=int, default=3, help="aux target layers (+1 verifier-last)")
    ap.add_argument("--min-len", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    hs_dir = out / "hs"
    hs_dir.mkdir(parents=True, exist_ok=True)
    n_layers = args.num_aux + 1

    input_ids_col, loss_mask_col, seq_len_col = [], [], []
    for i in range(args.n):
        seq = int(torch.randint(args.min_len, args.max_len + 1, (1,)).item())
        ids = torch.randint(0, args.vocab, (seq,), dtype=torch.long)
        mask = torch.zeros(seq, dtype=torch.bool)
        mask[seq // 2:] = True  # loss on the "response" half
        input_ids_col.append(ids.tolist())
        loss_mask_col.append(mask.tolist())
        seq_len_col.append(seq)
        save_file(
            {
                "hidden_states": torch.randn(seq, n_layers, args.hidden, dtype=torch.bfloat16),
                "token_ids": ids,
            },
            str(hs_dir / f"hs_{i}.safetensors"),
        )

    ds = Dataset.from_dict(
        {"input_ids": input_ids_col, "loss_mask": loss_mask_col, "seq_len": seq_len_col}
    )
    # ArrowDataset expects torch-formatted columns (data[i]["input_ids"] is a
    # Tensor), matching the real prep (preprocessing.py set_format(type="torch")).
    ds.set_format(type="torch")
    ds.save_to_disk(str(out / "arrow"))
    print(f"wrote {args.n} samples")
    print(f"  arrow: {out / 'arrow'}")
    print(f"  hs   : {hs_dir}  (hs_0..hs_{args.n - 1}.safetensors, [seq,{n_layers},{args.hidden}] bf16)")
    print(f"\ntrain with:  --data-path {out / 'arrow'}  --hidden-states-path {hs_dir}  --on-missing skip")


if __name__ == "__main__":
    main()
