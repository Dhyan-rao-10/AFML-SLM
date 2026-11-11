"""Pretraining batches sampled from a flat binary file of uint16 token ids."""

import os

import numpy as np
import torch


class TokenBins:
    def __init__(self, data_dir: str, block_size: int, device: str):
        self.data_dir = data_dir
        self.block_size = block_size
        self.device = device

    def _tokens(self, split: str):
        # Re-open the memmap per call: holding one open across a long run leaks memory.
        return np.memmap(os.path.join(self.data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")

    def num_tokens(self, split: str) -> int:
        return len(self._tokens(split))

    def _batch_at(self, data, offsets):
        T = self.block_size
        x = torch.from_numpy(np.stack([data[i : i + T].astype(np.int64) for i in offsets]))
        y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + T].astype(np.int64) for i in offsets]))
        return x, y

    def _to_device(self, x, y):
        if self.device.startswith("cuda"):
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def get_batch(self, split: str, batch_size: int):
        data = self._tokens(split)
        offsets = torch.randint(len(data) - self.block_size - 1, (batch_size,)).tolist()
        return self._to_device(*self._batch_at(data, offsets))

    def fixed_batches(self, split: str, n_batches: int, batch_size: int, seed: int = 1234):
        """The same evaluation batches every call and every run, so validation
        losses from different ablation runs are measured on identical tokens."""
        data = self._tokens(split)
        g = torch.Generator().manual_seed(seed)
        batches = []
        for _ in range(n_batches):
            offsets = torch.randint(len(data) - self.block_size - 1, (batch_size,), generator=g).tolist()
            batches.append(self._batch_at(data, offsets))
        return batches
