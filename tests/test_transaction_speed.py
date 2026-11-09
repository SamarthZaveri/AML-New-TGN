"""Regression checks for tensor live state and evaluation prefix replay."""
import copy
import unittest
from unittest import mock
import torch
from stage2.transaction_model import TransactionAccountTGN
from stage2.tbptt_memory import read_live


def dictionary_read(stored, ids, blocks):
    mapping = {}
    for keys, values in blocks:
        mapping.update(zip(keys.tolist(), values.unbind(0)))
    return torch.stack([mapping.get(k, stored[i]) for i, k in enumerate(ids.tolist())]) if len(ids) else stored


class TransactionSpeedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TransactionAccountTGN(mode='fused', num_account_nodes=12,
            num_bank_nodes=2, num_entity_nodes=2, account_feat_dim=3, bank_feat_dim=2,
            entity_feat_dim=2, edge_feat_dim=2, memory_dim=8, time_dim=4,
            num_attention_heads=2, dropout=0., enable_relay=True, max_neighbors=3)
        self.features = (torch.randn(12,3), torch.randn(2,2), torch.randn(2,2),
                         torch.arange(12)%2, torch.arange(12)%2)
        self.model.initialize(*self.features)
        self.batches = [dict(source_ids=torch.tensor([0,1,2,0]),
            dest_ids=torch.tensor([1,3,4,2]), edge_features=torch.randn(4,2),
            timestamps=torch.arange(4,dtype=torch.float64)+i*4, do_relay=True)
            for i in range(3)]

    def test_outputs_and_gradients_match_dictionary(self):
        reference = copy.deepcopy(self.model)
        outputs = []
        for model, reader in [(self.model, read_live), (reference, dictionary_read)]:
            with mock.patch('stage2.tbptt_memory.read_live', reader), mock.patch('stage2.transaction_model.read_live', reader):
                result = [model(**batch) for batch in self.batches]
                sum(o['final_account_logits'].sum()+o['transaction_logits'].sum() for o in result).backward()
                outputs.append(result)
        for a,b in zip(*outputs):
            for key in ('base_account_logits','final_account_logits','transaction_logits'):
                torch.testing.assert_close(a[key],b[key])
        for a,b in zip(self.model.parameters(),reference.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad)

    def test_prefix_preserves_predictions_and_checkpoint_state(self):
        for mode in ('fused','separate'):
            full = copy.deepcopy(self.model); fast = copy.deepcopy(self.model)
            for model in (full,fast):
                model.mode=mode; model.eval(); model.initialize(*self.features)
            with torch.no_grad():
                for batch in self.batches[:2]:
                    full(**batch); fast(**batch,score_accounts=False)
                a=full(**self.batches[2]); b=fast(**self.batches[2])
                for key in ('base_account_logits','final_account_logits','transaction_logits'):
                    torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                for key,value in full.state_dict().items():
                    torch.testing.assert_close(value,fast.state_dict()[key],rtol=0,atol=0)

    def test_empty_and_repeated_live_reads(self):
        stored=torch.zeros(4,2); ids=torch.tensor([2,1,2,7])
        first=torch.randn(2,2,requires_grad=True); last=torch.randn(1,2,requires_grad=True)
        blocks=[(torch.tensor([1,2]),first),(torch.tensor([2]),last)]
        torch.testing.assert_close(read_live(stored,ids,blocks),dictionary_read(stored,ids,blocks))
        self.assertEqual(read_live(stored[:0],ids[:0],blocks).shape,(0,2))

if __name__ == '__main__':
    unittest.main()
