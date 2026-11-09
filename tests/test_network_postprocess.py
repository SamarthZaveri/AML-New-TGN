import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stage2.network_postprocess import NetworkConfig, build_networks


class NetworkPostprocessTests(unittest.TestCase):
    def test_reconstructs_component_and_ordered_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({
                "sender_account_id": ["a", "b", "x"],
                "receiver_account_id": ["b", "c", "y"],
                "timestamp_unix": [0., 3600., 10.],
                "amount_paid": [100., 95., 50.],
                "amount_received": [100., 95., 50.],
            }).to_parquet(root / "events.parquet")
            pd.DataFrame({"account_id": ["a", "b", "c", "x", "y"],
                          "risk_score": [.9, .8, .7, .02, .01]}).to_parquet(root / "scores.parquet")
            result = build_networks(root / "events.parquet", root / "scores.parquet", root / "out",
                                    NetworkConfig(context_min_score=.5))
            self.assertEqual(result["candidate_events"], 2)
            paths = pd.read_parquet(root / "out" / "flow_paths.parquet")
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths.iloc[0].intermediary_account_id, "b")
            networks = pd.read_parquet(root / "out" / "networks.parquet")
            self.assertEqual(networks.iloc[0].account_count, 3)

