"""Export completed-history account scores from a trained Stage 2 checkpoint.

This is an offline readout for network post-processing. It replays the whole
event stream with the selected model weights and writes one score per account.
It is not the micro-batch inference service.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

import config
from stage2.checkpointing import load_trusted
from stage2.conservation_loss import ConservationLoss
from stage2.dataset import TemporalEventDataset
from stage2.model import MuleDetectorTGN
from stage2.train import replay


def export_scores(checkpoint: Path, events_file: Path, static_dir: Path,
                  chain_signals_file: Path, output: Path, batch_size: int,
                  device: str) -> None:
    checkpoint_data = load_trusted(checkpoint, "cpu")
    if checkpoint_data.get("kind") != "inference" or "weights" not in checkpoint_data:
        raise ValueError("Expected a trusted best.pt inference checkpoint")
    settings = checkpoint_data["settings"]
    dataset = TemporalEventDataset(events_file, static_dir, chain_signals_file, batch_size,
                                   split_mode=settings["split_mode"])
    if checkpoint_data["dataset_fingerprint"] != dataset.fingerprint:
        raise ValueError("Checkpoint and score-export dataset differ; use the original inputs and batch size")
    model = MuleDetectorTGN(**settings["model"]).to(device)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(checkpoint_data["weights"][name])
    conservation = ConservationLoss(settings["model"]["memory_dim"], dataset.num_accounts).to(device)
    stop = SimpleNamespace(requested=False)
    logits = replay(model, conservation, dataset, torch.device(device), settings["relay_interval"], stop)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"account_id": dataset.account_index.astype(str),
                  "risk_score": torch.sigmoid(logits).cpu().numpy(),
                  "mule_logit": logits.cpu().numpy()}).to_parquet(output, index=False)
    print(f"Wrote {len(dataset.account_index):,} account scores to {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=getattr(config, "BATCH_SIZE", 10_000))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--events-file", type=Path, default=config.EVENT_STREAM_FILE)
    p.add_argument("--static-dir", type=Path, default=config.SPARK_STATIC_DIR)
    p.add_argument("--chain-signals-file", type=Path, default=config.CHAIN_SIGNALS_FILE)
    args = p.parse_args()
    export_scores(args.checkpoint, args.events_file, args.static_dir, args.chain_signals_file,
                  args.output, args.batch_size, args.device)


if __name__ == "__main__":
    main()
