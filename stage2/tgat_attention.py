"""Temporal attention and deterministic, serializable recent-event history."""
import math
import torch
from torch import nn
from stage2.time_encoding import Time2Vec


class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, memory_dim, time_dim, num_heads=4, dropout=0.1):
        super().__init__()
        if memory_dim % num_heads:
            raise ValueError('memory_dim must be divisible by num_heads')
        self.memory_dim, self.time_dim = memory_dim, time_dim
        self.num_heads, self.head_dim = num_heads, memory_dim // num_heads
        self.time_encoder = Time2Vec(time_dim)
        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim + time_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim + time_dim, memory_dim)
        self.output_proj = nn.Linear(memory_dim, memory_dim)
        self.layer_norm = nn.LayerNorm(memory_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_memory, neighbor_memories, neighbor_time_deltas, neighbor_mask=None):
        b, k, _ = neighbor_memories.shape
        enc = self.time_encoder(neighbor_time_deltas).reshape(b, k, self.time_dim)
        nb = torch.cat((neighbor_memories, enc), -1)
        def heads(x, n):
            return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        q = heads(self.query_proj(target_memory), 1)
        key, val = heads(self.key_proj(nb), k), heads(self.value_proj(nb), k)
        scores = (q @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        valid = (torch.ones(b, k, dtype=torch.bool, device=scores.device)
                 if neighbor_mask is None else neighbor_mask)
        mask = valid[:, None, None, :]
        # Finite sentinel works in fp32/bfloat16/fp16; empty rows become zero.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(-1) * mask.to(scores.dtype)
        ctx = (self.dropout(weights) @ val).squeeze(2).reshape(b, self.memory_dim)
        projected = self.dropout(self.output_proj(ctx))
        projected = projected * valid.any(-1, keepdim=True)
        return self.layer_norm(target_memory + projected)


class NeighborSampler(nn.Module):
    """Last K incident events, including repeated neighbors. All buffers checkpointed."""
    def __init__(self, num_nodes, max_neighbors=20):
        super().__init__()
        if max_neighbors < 1:
            raise ValueError('max_neighbors must be positive')
        self.max_neighbors = max_neighbors
        self.register_buffer('neighbor_ids', torch.full((num_nodes, max_neighbors), -1, dtype=torch.long))
        self.register_buffer('neighbor_times', torch.zeros(num_nodes, max_neighbors, dtype=torch.float64))
        self.register_buffer('write_ptr', torch.zeros(num_nodes, dtype=torch.long))

    @torch.no_grad()
    def update(self, source_ids, dest_ids, timestamps, edge_features=None):
        # Input events sorted; interleave both directions before grouping.
        owners = torch.stack((dest_ids, source_ids), 1).flatten()
        neighbors = torch.stack((source_ids, dest_ids), 1).flatten()
        times = timestamps.repeat_interleave(2)
        if not owners.numel():
            return
        order = torch.argsort(owners, stable=True)
        owner, neighbor, time = owners[order], neighbors[order], times[order]
        unique, counts = torch.unique_consecutive(owner, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(len(owner), device=owner.device) - starts.repeat_interleave(counts)
        # Discard overwritten entries BEFORE scatter, so each output index is unique.
        keep = rank >= (counts - self.max_neighbors).clamp(min=0).repeat_interleave(counts)
        pos = (self.write_ptr[owner[keep]] + rank[keep]) % self.max_neighbors
        flat = owner[keep] * self.max_neighbors + pos
        self.neighbor_ids.view(-1)[flat] = neighbor[keep]
        self.neighbor_times.view(-1)[flat] = time[keep].to(self.neighbor_times.dtype)
        self.write_ptr[unique] = (self.write_ptr[unique] + counts) % self.max_neighbors

    def get_neighbors(self, node_ids, current_time, memory, device=None):
        ids, times = self.neighbor_ids[node_ids], self.neighbor_times[node_ids]
        valid = (ids >= 0) & (times <= current_time)
        values = memory[ids.clamp(min=0)] * valid.unsqueeze(-1)
        deltas = (current_time - times).clamp(min=0) * valid
        return values, deltas, valid

    @torch.no_grad()
    def reset(self):
        self.neighbor_ids.fill_(-1)
        self.neighbor_times.zero_()
        self.write_ptr.zero_()
