"""Optional flow and chain objectives. Unknown chain labels are not negatives."""
import torch
from torch import nn
from torch.nn import functional as F


class ConservationLoss(nn.Module):
    def __init__(self, memory_dim, num_accounts, window_hours=72., eps=1e-8, gap_weight=0.1):
        super().__init__()
        if window_hours <= 0:
            raise ValueError('window_hours must be positive')
        self.window, self.eps, self.gap_weight = window_hours, eps, gap_weight
        self.flow_head = nn.Sequential(nn.Linear(memory_dim, memory_dim//2), nn.ReLU(), nn.Linear(memory_dim//2, 2))
        self.register_buffer('in_flow', torch.zeros(num_accounts))
        self.register_buffer('out_flow', torch.zeros(num_accounts))
        self.register_buffer('last_update', torch.zeros(num_accounts, dtype=torch.float64))
        self.register_buffer('seen', torch.zeros(num_accounts, dtype=torch.bool))
        self.register_buffer('target_mean', torch.tensor(0.))
        self.register_buffer('target_var', torch.tensor(1.))
        self.register_buffer('target_n', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def reset_state(self):
        self.in_flow.zero_()
        self.out_flow.zero_()
        self.last_update.zero_()
        self.seen.zero_()

    @torch.no_grad()
    def update_flows(self, source_ids, dest_ids, amounts_paid, amounts_received, timestamps):
        now = timestamps.max()
        ids = torch.unique(torch.cat((source_ids, dest_ids)))
        decay = torch.where(self.seen[ids], torch.exp(-(now - self.last_update[ids]).clamp(min=0)/self.window), 0.)
        # Both sides decayed to the same cutoff exactly once per touched account.
        self.in_flow[ids] *= decay.to(self.in_flow.dtype)
        self.out_flow[ids] *= decay.to(self.out_flow.dtype)
        # Discount each contribution to batch end, rather than treating the batch as instantaneous.
        event_decay = torch.exp(-(now - timestamps)/self.window).to(amounts_paid.dtype)
        self.out_flow.index_add_(0, source_ids, amounts_paid * event_decay)
        self.in_flow.index_add_(0, dest_ids, amounts_received * event_decay)
        self.last_update[ids] = now.to(self.last_update.dtype)
        self.seen[ids] = True

    def forward(self, embeddings, node_ids, source_ids, dest_ids,
                amounts_paid, amounts_received, timestamps, supervision_mask=None):
        self.update_flows(source_ids, dest_ids, amounts_paid, amounts_received, timestamps)
        mask = (torch.ones_like(node_ids, dtype=torch.bool) if supervision_mask is None else supervision_mask)
        if not mask.any():
            return embeddings.sum() * 0
        ids = node_ids[mask]
        targets = torch.stack((torch.log1p(self.in_flow[ids]), torch.log1p(self.out_flow[ids])), -1)
        with torch.no_grad():
            mean, var = targets.mean(), targets.var(unbiased=False).clamp(min=self.eps)
            if self.target_n == 0:
                self.target_mean.copy_(mean)
                self.target_var.copy_(var)
            else:
                self.target_mean.lerp_(mean, .01)
                self.target_var.lerp_(var, .01)
            self.target_n.add_(1)
        targets = (targets - self.target_mean) / self.target_var.sqrt().clamp(min=self.eps)
        pred = self.flow_head(embeddings[mask])
        reconstruction = F.mse_loss(pred, targets)
        gap = F.smooth_l1_loss(pred[:, 0], pred[:, 1])
        return reconstruction + self.gap_weight * gap


class FocalLoss(nn.Module):
    def __init__(self, gamma=2., pos_weight=None):
        super().__init__()
        self.gamma, self.pos_weight = gamma, pos_weight

    def forward(self, logits, targets):
        if logits.numel() == 0:
            return logits.sum() * 0
        y = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        p = logits.sigmoid()
        pt = y*p + (1-y)*(1-p)
        w = torch.ones_like(y) if self.pos_weight is None else 1 + (self.pos_weight-1)*y
        return (w * (1-pt).pow(self.gamma) * bce).sum() / w.sum().clamp(min=1)


class ChainAuxiliaryLoss(nn.Module):
    def forward(self, chain_membership_logits, chain_score_preds, chain_membership_targets,
                chain_score_targets, known_mask, score_mask):
        # A positive-only file cannot train binary membership; score regression remains valid.
        known_y = chain_membership_targets[known_mask]
        both_classes = known_y.numel() > 0  # caller disables membership unless training labels contain both classes
        membership = (F.binary_cross_entropy_with_logits(chain_membership_logits[known_mask], known_y.float())
                      if both_classes else chain_membership_logits.sum()*0)
        score = (F.mse_loss(chain_score_preds[score_mask], chain_score_targets[score_mask])
                 if score_mask.any() else chain_score_preds.sum()*0)
        return membership + score
