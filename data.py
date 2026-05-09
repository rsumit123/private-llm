import glob
import os
import numpy as np
import torch


class ShardDataset:
    """Memory-maps all .bin shards in a directory; yields random fixed-length sequences."""
    def __init__(self, data_dir, seq_len, dtype=np.uint16, seed=0):
        self.seq_len = seq_len
        self.dtype = dtype
        files = sorted(glob.glob(os.path.join(data_dir, "shard_*.bin")))
        if not files:
            raise FileNotFoundError(f"no shard_*.bin in {data_dir}")
        self.shards = [np.memmap(f, dtype=dtype, mode="r") for f in files]
        self.shard_lens = [len(s) for s in self.shards]
        self.total = sum(self.shard_lens)
        self.rng = np.random.default_rng(seed)
        print(f"loaded {len(files)} shards, {self.total:,} tokens")

    def get_batch(self, batch_size, device):
        # sample batch_size random (shard, offset) pairs
        sx = self.rng.integers(0, len(self.shards), size=batch_size)
        out_x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        out_y = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for i, s in enumerate(sx):
            shard = self.shards[s]
            off = self.rng.integers(0, len(shard) - self.seq_len - 1)
            chunk = shard[off:off + self.seq_len + 1].astype(np.int64)
            out_x[i] = chunk[:-1]
            out_y[i] = chunk[1:]
        x = torch.from_numpy(out_x).to(device, non_blocking=True)
        y = torch.from_numpy(out_y).to(device, non_blocking=True)
        return x, y
