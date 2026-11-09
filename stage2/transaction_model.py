"""Separate-head late fusion versus differentiable transaction-informed account fusion."""
import torch
from torch import nn
from stage2.model import MuleDetectorTGN
from stage2.tbptt_memory import WindowMemory, read_live
from stage2.tgn_memory import elapsed

class TransactionAccountTGN(MuleDetectorTGN):
    def __init__(self, mode='separate', **kwargs):
        super().__init__(**kwargs)
        if mode not in ('separate','fused'): raise ValueError('Unknown experiment mode')
        self.mode = mode
        names = ('num_account_nodes','num_bank_nodes','num_entity_nodes','account_feat_dim',
                 'bank_feat_dim','entity_feat_dim','edge_feat_dim','memory_dim','time_dim')
        self.tgn_memory = WindowMemory(**{k:kwargs[k] for k in names})
        d = kwargs['memory_dim']
        self.transaction_head = nn.Sequential(nn.Linear(2*d+kwargs['edge_feat_dim']+2*kwargs['time_dim'],d),
                                   nn.SiLU(),nn.Dropout(kwargs.get('dropout',.1)),nn.Linear(d,1))
        self.account_transaction_head = nn.Sequential(nn.Linear(d+2,d),nn.SiLU(),nn.Linear(d,1))
        if mode=='separate':
            for p in self.account_transaction_head.parameters(): p.requires_grad_(False)
        # Auxiliary legacy heads are unused in these experiments.
        for head in (self.chain_membership_head,self.chain_score_head):
            for p in head.parameters(): p.requires_grad_(False)
        n=kwargs['num_account_nodes']
        self.register_buffer('risk_sum',torch.zeros(n))
        self.register_buffer('risk_count',torch.zeros(n))
        self.risk_live=[]

    def initialize(self,*args):
        super().initialize(*args)
        self.risk_sum.zero_(); self.risk_count.zero_(); self.risk_live=[]

    def detach_window(self):
        self.tgn_memory.detach_window(); self.risk_live=[]

    def _score(self, ids, memories, current_time):
        # Read live sparse neighbor states when inside a TBPTT window.
        sampler=self.neighbor_sampler
        nb=sampler.neighbor_ids[ids]; times=sampler.neighbor_times[ids]
        valid=(nb>=0)&(times<=current_time)
        values=self.tgn_memory.read('account',nb.clamp(min=0).flatten()).reshape(len(ids),sampler.max_neighbors,-1)
        attended=self.tgat_attention(memories,values*valid[:,:,None],(current_time-times).clamp(min=0)*valid,valid)
        static=self.static_encoder(self.tgn_memory.account_features[ids])
        count=self.tgn_memory.account_event_count[ids]
        h=self.fuse(torch.cat([attended,static,torch.log1p(count.float())[:,None]/5],-1))
        return {'mule_logits':self.classifier(h).squeeze(-1),'embeddings':h,'node_ids':ids}

    def forward(self,source_ids,dest_ids,edge_features,timestamps,do_relay=False,merge_alpha=.5,score_accounts=True):
        mem=self.tgn_memory
        src=mem.read('account',source_ids)+mem.account_proj(mem.account_features[source_ids])
        dst=mem.read('account',dest_ids)+mem.account_proj(mem.account_features[dest_ids])
        ts=elapsed(timestamps,mem.account_last_update[source_ids],mem.account_seen[source_ids])
        td=elapsed(timestamps,mem.account_last_update[dest_ids],mem.account_seen[dest_ids])
        features=torch.cat([src,dst,edge_features,mem.time_encoder(ts),mem.time_encoder(td)],-1)
        # Experiment 1 learns the transaction head independently of backbone gradients.
        transaction_logits=self.transaction_head(features.detach() if self.mode=='separate' else features).squeeze(-1)
        prior_ids=torch.unique(torch.cat([source_ids,dest_ids]));prior=mem.account_event_count[prior_ids].clone()
        ids,live=mem.process_event_batch(source_ids,dest_ids,edge_features,timestamps)
        now=float(timestamps.max())
        if self.enable_relay and do_relay: live=mem.relay_batch(ids,live,self.account_bank,self.account_entity,now)
        self.neighbor_sampler.update(source_ids,dest_ids,timestamps)
        out=self._score(ids,live,now) if score_accounts else {'node_ids':ids}
        positions=torch.searchsorted(ids,torch.cat([source_ids,dest_ids]))
        probabilities=transaction_logits.sigmoid()
        sums=probabilities.new_zeros(len(ids)).index_add(0,positions,probabilities.repeat(2))
        counts=torch.bincount(positions,minlength=len(ids)).to(sums.dtype)
        old=self.risk_sum[ids]
        if torch.is_grad_enabled() and self.risk_live:
            old=read_live(old,ids,self.risk_live)
        cumulative=old+sums; total=self.risk_count[ids]+counts
        average=cumulative/total.clamp(min=1)
        if torch.is_grad_enabled():self.risk_live.append((ids,cumulative))
        with torch.no_grad():self.risk_sum[ids]=cumulative.detach();self.risk_count[ids]=total
        if not score_accounts:
            out.update(transaction_logits=transaction_logits,transaction_risk_mean=average,
                       prior_event_counts=prior)
            return out
        base=out['mule_logits']
        if self.mode=='fused':
            final=self.account_transaction_head(torch.cat([out['embeddings'],average[:,None],torch.log1p(total)[:,None]/5],-1)).squeeze(-1)
        else:
            final=torch.logit(((1-merge_alpha)*base.sigmoid()+merge_alpha*average).clamp(1e-6,1-1e-6))
        out.update(base_account_logits=base,final_account_logits=final,transaction_logits=transaction_logits,
                   transaction_risk_mean=average,prior_event_counts=prior)
        return out
