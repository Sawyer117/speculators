from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    SampleFileDataset,
    create_collate_fn,
    split_files,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


def _setup_dataloader(
    dataset: BaseDataset,
    total_seq_len: int,
    hidden_size: int,
    num_workers: int = 12,
    num_target_layers: int = 3,
    prefetch_factor: int | None = 4,
    preprocess: Callable[[BatchType], BatchType] | None = None,
) -> DataLoader:
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
    )
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=True,
        collate_fn=create_collate_fn(
            total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            dtype=dataset.hidden_states_dtype,
            preprocess=preprocess,
        ),
        persistent_workers=use_workers,
    )


def create_train_val_loaders(
    *,
    data_path: str,
    train_data_ratio: float,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    noise_std: float,
    legacy_data: bool,
    hidden_states_path: str | None,
    vllm_endpoint: str,
    on_missing: Literal["generate", "skip", "warn", "raise"],
    on_generate: Literal["cache", "delete"],
    verifier_name_or_path: str,
    served_model_name: str | None = None,
    request_timeout: float | None,
    max_retries: int,
    hidden_size: int,
    num_target_layers: int,
    num_workers: int,
    prefetch_factor: int,
    preprocess: Callable[[BatchType], BatchType] | None,
    no_validation: bool = False,
) -> tuple[DataLoader, DataLoader | None]:
    """Create training and validation DataLoaders.

    Handles dataset construction (legacy vs Arrow) and dataloader wiring.
    Non-data SP ranks get lightweight loaders with no workers (they receive
    batches via scatter).  Reads DP/SP topology from
    :mod:`speculators.train.distributed`.
    """
    noise_transform = AddUniformNoise(std=noise_std)

    # --no-validation: skip the per-epoch val pass and train on the FULL dataset
    # (split_ratio 1.0). The Trainer already skips validation when val_loader is None;
    # reclaiming the held-out slice avoids silently wasting 1 - train_data_ratio of the data.
    if not no_validation and not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")
    train_split_ratio = 1.0 if no_validation else train_data_ratio
    val_dataset: BaseDataset | None = None

    if legacy_data:
        warnings.warn(
            "Using '--legacy-data' is deprecated and will be removed soon.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        train_files, val_files = split_files(data_path, ratio=train_data_ratio)
        train_dataset: BaseDataset = SampleFileDataset(
            file_list=train_files,
            max_len=total_seq_len,
            transform=noise_transform,
            hidden_states_dtype=hidden_states_dtype,
        )
        if not no_validation:
            val_dataset = SampleFileDataset(
                file_list=val_files,
                max_len=total_seq_len,
                hidden_states_dtype=hidden_states_dtype,
            )
    else:
        train_dataset = ArrowDataset(
            datapath=data_path,
            max_len=total_seq_len,
            hidden_states_path=hidden_states_path,
            vllm_endpoint=vllm_endpoint,
            on_missing=on_missing,
            on_generate=on_generate,
            transform=noise_transform,
            split_ratio=train_split_ratio,
            model=served_model_name or verifier_name_or_path,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
        if not no_validation:
            val_dataset = ArrowDataset(
                datapath=data_path,
                max_len=total_seq_len,
                hidden_states_path=hidden_states_path,
                vllm_endpoint=vllm_endpoint,
                on_missing=on_missing,
                on_generate=on_generate,
                split_ratio=train_data_ratio - 1.0,
                model=served_model_name or verifier_name_or_path,
                hidden_states_dtype=hidden_states_dtype,
                request_timeout=request_timeout,
                max_retries=max_retries,
            )

    train_loader = _setup_dataloader(
        train_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
    )
    # NB: the VAL loader forces num_workers=0 (no forked workers). Unlike the train loader —
    # whose persistent workers fork ONCE at start, when the NPU/HCCL context is fresh — the val
    # loader's first iteration is at the epoch BOUNDARY, i.e. after a full epoch of NPU use + the
    # (EP-)DCP checkpoint gather. Forking dataloader workers from that live native state corrupts
    # the child heap → "free(): invalid pointer" + "DataLoader worker killed by signal: Aborted",
    # and the run dies right after epoch 0's checkpoint save. Validation is a small held-out pass,
    # so in-process loading costs ~nothing (the trainer is HS-fetch-bound anyway; fetch_frac≈0.02).
    val_loader = None
    if not no_validation:
        val_loader = _setup_dataloader(
            val_dataset,
            total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            num_workers=0,
            prefetch_factor=None,
            preprocess=preprocess,
        )

    return train_loader, val_loader
