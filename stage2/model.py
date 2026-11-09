"""TGN + TGAT account classifier. Shared attention/fusion/readout for all scoring."""
import torch
from torch import nn
from stage2.tgn_memory import TGNMemory
from stage2.tgat_attention import TemporalMultiHeadAttention, NeighborSampler


class MuleDetectorTGN(nn.Module):
    def __init__(self, num_account_nodes, num_bank_nodes, num_entity_nodes,
                 account_feat_dim, bank_feat_dim, entity_feat_dim, edge_feat_dim,
                 memory_dim, time_dim, num_attention_heads=4, dropout=0.1,
                 enable_relay=True, max_neighbors=20):
        super().__init__()
        self.enable_relay = enable_relay
        d = self.memory_dim = memory_dim
        self.tgn_memory = TGNMemory(num_account_nodes, num_bank_nodes, num_entity_nodes,
                                   account_feat_dim, bank_feat_dim, entity_feat_dim,
                                   edge_feat_dim, d, time_dim)
        self.neighbor_sampler = NeighborSampler(num_account_nodes, max_neighbors)
        self.tgat_attention = TemporalMultiHeadAttention(d, time_dim, num_attention_heads, dropout)
        self.static_encoder = nn.Sequential(nn.Linear(account_feat_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.fuse = nn.Sequential(nn.Linear(2*d + 1, d), nn.SiLU(), nn.LayerNorm(d))
        self.classifier = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Dropout(dropout),
                                        nn.Linear(d, d//2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d//2, 1))
        self.chain_membership_head = nn.Linear(d, 1)
        self.chain_score_head = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        self.register_buffer('account_bank', torch.full((num_account_nodes,), -1, dtype=torch.long), persistent=False)
        self.register_buffer('account_entity', torch.full((num_account_nodes,), -1, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def initialize(self, account_features, bank_features, entity_features, account_bank, account_entity):
        self.tgn_memory.initialize_memory(account_features, bank_features, entity_features)
        self.account_bank.copy_(account_bank)
        self.account_entity.copy_(account_entity)
        self.neighbor_sampler.reset()

    def reset_state(self):
        self.tgn_memory.reset_memory()
        self.neighbor_sampler.reset()

    def _score(self, ids, memories, current_time):
        nm, nd, mask = self.neighbor_sampler.get_neighbors(ids, current_time, self.tgn_memory.account_memory)
        attended = self.tgat_attention(memories, nm, nd, mask)
        static = self.static_encoder(self.tgn_memory.account_features[ids])
        count = self.tgn_memory.account_event_count[ids]
        h = self.fuse(torch.cat((attended, static, torch.log1p(count.float()).unsqueeze(-1)/5), -1))
        return {'mule_logits': self.classifier(h).squeeze(-1),
                'chain_membership_logits': self.chain_membership_head(h).squeeze(-1),
                'chain_score_preds': self.chain_score_head(h).squeeze(-1),
                'embeddings': h, 'node_ids': ids, 'event_counts': count}

    def forward(self, source_ids, dest_ids, edge_features, timestamps, do_relay=False):
        ids = torch.unique(torch.cat((source_ids, dest_ids)))
        # Prior history means BEFORE this batch, not including its events.
        prior = self.tgn_memory.account_event_count[ids].clone()
        ids, live = self.tgn_memory.process_event_batch(source_ids, dest_ids, edge_features, timestamps)
        now = float(timestamps.max().item())
        if self.enable_relay and do_relay:
            live = self.tgn_memory.relay_batch(ids, live, self.account_bank, self.account_entity, now)
        # Batch-end prediction: current batch's neighbors are already observed.
        self.neighbor_sampler.update(source_ids, dest_ids, timestamps)
        out = self._score(ids, live, now)
        out['prior_event_counts'] = prior
        return out

    @torch.no_grad()
    def predict_accounts(self, account_ids, current_time, chunk=4096):
        if self.training:
            raise RuntimeError('Call model.eval() before inference')
        if chunk < 1:
            raise ValueError('chunk must be positive')
        keys = ('mule_logits', 'chain_membership_logits', 'chain_score_preds')
        outputs = {k: [] for k in keys}
        for ids in account_ids.split(chunk):
            if not len(ids):
                continue
            out = self._score(ids, self.tgn_memory.account_memory[ids], current_time)
            for k in keys:
                outputs[k].append(out[k])
        return {k: torch.cat(v) if v else self.account_bank.new_empty(0, dtype=torch.float32)
                for k, v in outputs.items()}
