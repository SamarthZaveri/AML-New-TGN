"""Batch-end TGN with local differentiable relay and explicit first-seen clocks.
Persistent history is detached; same-batch transaction/relay tensors retain gradients.
"""
import torch
from torch import nn
from stage2.time_encoding import Time2Vec


class MessageComputer(nn.Module):
    def __init__(self, d, edge_dim, time_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*d + edge_dim + time_dim, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, source, dest, edge, time):
        return self.net(torch.cat((source, dest, edge, time), -1))


def elapsed(timestamps, last, seen):
    return torch.where(seen, (timestamps - last).clamp(min=0), torch.zeros_like(timestamps))


def aggregate_update(memory, gru, ids, messages, timestamps, last, seen):
    unique, inv = torch.unique(ids, return_inverse=True)
    agg = messages.new_zeros(len(unique), messages.size(-1))
    agg.index_add_(0, inv, messages)
    counts = torch.bincount(inv, minlength=len(unique)).to(messages.dtype)
    new = gru(agg / counts.unsqueeze(-1), memory[unique])
    # Advanced-index reads above are copies; persistent writes carry no graph.
    with torch.no_grad():
        memory[unique] = new.detach()
        latest = timestamps.new_full((len(unique),), -torch.inf)
        latest.scatter_reduce_(0, inv, timestamps, reduce='amax', include_self=True)
        last[unique] = latest.to(last.dtype)
        seen[unique] = True
    return unique, new


class TGNMemory(nn.Module):
    def __init__(self, num_account_nodes, num_bank_nodes, num_entity_nodes,
                 account_feat_dim, bank_feat_dim, entity_feat_dim,
                 edge_feat_dim, memory_dim, time_dim):
        super().__init__()
        self.memory_dim = memory_dim
        d = memory_dim
        self.time_encoder = Time2Vec(time_dim)
        # Separate direction networks preserve sender/receiver roles.
        self.msg_account_txn_in = MessageComputer(d, edge_feat_dim, time_dim)
        self.msg_account_txn_out = MessageComputer(d, edge_feat_dim, time_dim)
        self.msg_account_to_bank = MessageComputer(d, 0, time_dim)
        self.msg_bank_to_account = MessageComputer(d, 0, time_dim)
        self.msg_account_to_entity = MessageComputer(d, 0, time_dim)
        self.msg_entity_to_account = MessageComputer(d, 0, time_dim)
        self.account_gru = nn.GRUCell(d, d)
        self.bank_gru = nn.GRUCell(d, d)
        self.entity_gru = nn.GRUCell(d, d)
        self.relay_gru = nn.GRUCell(d, d)
        for kind, n, features in [('account', num_account_nodes, account_feat_dim),
                                  ('bank', num_bank_nodes, bank_feat_dim),
                                  ('entity', num_entity_nodes, entity_feat_dim)]:
            setattr(self, kind + '_proj', nn.Linear(features, d))
            self.register_buffer(kind + '_features', torch.zeros(n, features), persistent=False)
            self.register_buffer(kind + '_memory', torch.zeros(n, d))
            self.register_buffer(kind + '_last_update', torch.zeros(n, dtype=torch.float64))
            self.register_buffer(kind + '_seen', torch.zeros(n, dtype=torch.bool))
        self.register_buffer('account_event_count', torch.zeros(num_account_nodes, dtype=torch.long))

    @torch.no_grad()
    def initialize_memory(self, account_features, bank_features, entity_features):
        for kind, value in [('account', account_features), ('bank', bank_features), ('entity', entity_features)]:
            getattr(self, kind + '_features').copy_(value)
        self.reset_memory()

    @torch.no_grad()
    def reset_memory(self):
        for kind in ('account', 'bank', 'entity'):
            for suffix in ('_memory', '_last_update', '_seen'):
                getattr(self, kind + suffix).zero_()
        self.account_event_count.zero_()

    def get_memory(self, node_ids):
        return self.account_memory[node_ids]

    def process_event_batch(self, source_ids, dest_ids, edge_features, timestamps):
        source = self.account_memory[source_ids] + self.account_proj(self.account_features[source_ids])
        dest = self.account_memory[dest_ids] + self.account_proj(self.account_features[dest_ids])
        td = elapsed(timestamps, self.account_last_update[dest_ids], self.account_seen[dest_ids])
        ts = elapsed(timestamps, self.account_last_update[source_ids], self.account_seen[source_ids])
        incoming = self.msg_account_txn_in(source, dest, edge_features, self.time_encoder(td))
        outgoing = self.msg_account_txn_out(dest, source, edge_features, self.time_encoder(ts))
        all_ids = torch.cat((dest_ids, source_ids))
        ids, new = aggregate_update(self.account_memory, self.account_gru, all_ids,
                                    torch.cat((incoming, outgoing)), timestamps.repeat(2),
                                    self.account_last_update, self.account_seen)
        with torch.no_grad():
            self.account_event_count.index_add_(0, all_ids, torch.ones_like(all_ids))
        return ids, new

    def relay_batch(self, account_ids, account_mem, account_bank, account_entity, current_time):
        ctx, count = torch.zeros_like(account_mem), account_mem.new_zeros(len(account_ids), 1)
        for kind, mapping in [('bank', account_bank), ('entity', account_entity)]:
            all_hubs = mapping[account_ids]
            ai = (all_hubs >= 0).nonzero(as_tuple=True)[0]
            if not ai.numel():
                continue
            hi, am = all_hubs[ai], account_mem[ai]
            memory = getattr(self, kind + '_memory')
            last, seen = getattr(self, kind + '_last_update'), getattr(self, kind + '_seen')
            proj = getattr(self, kind + '_proj')(getattr(self, kind + '_features')[hi])
            t = torch.full((len(ai),), current_time, device=am.device, dtype=torch.float64)
            empty = am.new_empty(len(ai), 0)
            up = getattr(self, 'msg_account_to_' + kind)(am, memory[hi] + proj, empty,
                                                       self.time_encoder(elapsed(t, last[hi], seen[hi])))
            ui, updated = aggregate_update(memory, getattr(self, kind + '_gru'), hi, up, t, last, seen)
            live_hub = updated[torch.searchsorted(ui, hi)]
            down = getattr(self, 'msg_' + kind + '_to_account')(
                live_hub, am, empty, self.time_encoder(t - self.account_last_update[account_ids[ai]]))
            ctx = ctx.index_add(0, ai, down)
            count.index_add_(0, ai, torch.ones_like(count[ai]))
        if not count.any():
            return account_mem
        new = self.relay_gru(ctx / count.clamp(min=1), account_mem)
        new = torch.where(count > 0, new, account_mem)
        with torch.no_grad():
            self.account_memory[account_ids] = new.detach()
        # Transaction clocks deliberately remain unchanged by relay.
        return new
