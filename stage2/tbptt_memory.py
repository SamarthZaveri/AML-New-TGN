"""Sparse live states retain gradients inside a bounded optimizer window.
Detached dense buffers preserve history outside the window. Existing TGN is unchanged.
"""
import torch
from stage2.tgn_memory import TGNMemory, elapsed

def read_live(stored, ids, blocks):
    """Overlay sparse, differentiable writes in order; latest write wins."""
    for keys, values in blocks:
        if not keys.numel():
            continue
        positions = torch.searchsorted(keys, ids).clamp(max=len(keys)-1)
        matched = keys[positions] == ids
        rows = matched.nonzero(as_tuple=True)[0]
        stored = stored.index_copy(0, rows, values[positions[rows]])
    return stored


class WindowMemory(TGNMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live = {k: [] for k in ('account','bank','entity')}

    def detach_window(self):
        self.live = {k: [] for k in ('account','bank','entity')}

    def reset_memory(self):
        super().reset_memory()
        self.detach_window()

    def read(self, kind, ids):
        ids = ids.reshape(-1)
        stored = getattr(self, kind+'_memory')[ids]
        if not torch.is_grad_enabled() or not self.live[kind]: return stored
        return read_live(stored, ids, self.live[kind])

    def store(self, kind, ids, values):
        if torch.is_grad_enabled():
            order = torch.argsort(ids)
            self.live[kind].append((ids[order], values[order]))
        with torch.no_grad(): getattr(self,kind+'_memory')[ids] = values.detach()

    def update(self, kind, ids, messages, timestamps, gru):
        unique,inv = torch.unique(ids, return_inverse=True)
        sums = messages.new_zeros(len(unique),messages.size(-1)).index_add(0,inv,messages)
        counts = torch.bincount(inv,minlength=len(unique)).to(messages.dtype)
        values = gru(sums/counts[:,None], self.read(kind,unique))
        self.store(kind,unique,values)
        with torch.no_grad():
            latest = timestamps.new_full((len(unique),),-torch.inf)
            latest.scatter_reduce_(0,inv,timestamps,reduce='amax',include_self=True)
            getattr(self,kind+'_last_update')[unique] = latest
            getattr(self,kind+'_seen')[unique] = True
        return unique,values

    def process_event_batch(self, source_ids, dest_ids, edge_features, timestamps):
        source = self.read('account',source_ids) + self.account_proj(self.account_features[source_ids])
        dest = self.read('account',dest_ids) + self.account_proj(self.account_features[dest_ids])
        ts = elapsed(timestamps,self.account_last_update[source_ids],self.account_seen[source_ids])
        td = elapsed(timestamps,self.account_last_update[dest_ids],self.account_seen[dest_ids])
        incoming = self.msg_account_txn_in(source,dest,edge_features,self.time_encoder(td))
        outgoing = self.msg_account_txn_out(dest,source,edge_features,self.time_encoder(ts))
        endpoints = torch.cat([dest_ids,source_ids])
        ids,values = self.update('account',endpoints,torch.cat([incoming,outgoing]),timestamps.repeat(2),self.account_gru)
        with torch.no_grad(): self.account_event_count.index_add_(0,endpoints,torch.ones_like(endpoints))
        return ids,values

    def relay_batch(self, ids, values, bank_map, entity_map, current_time):
        context = torch.zeros_like(values); count = values.new_zeros(len(ids),1)
        for kind,mapping in [('bank',bank_map),('entity',entity_map)]:
            hubs = mapping[ids]; active = (hubs>=0).nonzero(as_tuple=True)[0]
            if not len(active): continue
            hub_ids,account = hubs[active],values[active]
            prior = self.read(kind,hub_ids) + getattr(self,kind+'_proj')(getattr(self,kind+'_features')[hub_ids])
            time = torch.full((len(active),),current_time,device=values.device,dtype=torch.float64)
            delta = elapsed(time,getattr(self,kind+'_last_update')[hub_ids],getattr(self,kind+'_seen')[hub_ids])
            empty = account.new_empty(len(active),0)
            message = getattr(self,'msg_account_to_'+kind)(account,prior,empty,self.time_encoder(delta))
            unique,live = self.update(kind,hub_ids,message,time,getattr(self,kind+'_gru'))
            down = getattr(self,'msg_'+kind+'_to_account')(live[torch.searchsorted(unique,hub_ids)],account,empty,
                         self.time_encoder(time-self.account_last_update[ids[active]]))
            context = context.index_add(0,active,down)
            count.index_add_(0,active,torch.ones_like(count[active]))
        updated = torch.where(count>0,self.relay_gru(context/count.clamp(min=1),values),values)
        self.store('account',ids,updated)
        return updated
