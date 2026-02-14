import tiktoken
import torch
import numpy as np
import os

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt,dtype = torch.long)
    return ptt

class Dataloaderlite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in ['train', 'val']

        data_root = "edu_fineweb10B"
        self.shards = os.listdir(data_root)
        self.shards = [s for s in self.shards if split in s]
        self.shards = sorted(self.shards)
        self.shards = [os.path.join(data_root, s) for s in self.shards]
        assert len(self.shards) > 0, f"No shards found for split {split}"

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position + B*T + 1]

        x = buf[:-1].view(B, T).contiguous()
        y = buf[1:].view(B, T).contiguous()

        self.current_position += B * T * self.num_processes

        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y
