# Stage 2 Offline Network Post-processing Delivery

Copy the files in this archive into the existing project root while preserving
their paths. It adds:

- `stage2/export_scores.py`: creates `account_scores.parquet` from `best.pt`.
- `stage2/network_postprocess.py`: builds offline suspicious networks, edges,
  and time/amount-consistent flow paths.
- `docs/stage2/NETWORK_POSTPROCESSING.md`: commands, output contract, and
  network-score definition.
- `tests/test_network_postprocess.py`: synthetic reconstruction test.

The score export replays the full completed history. The network layer then
uses that completed graph for retrospective investigation. Neither module is
the online inference engine or shares its mutable temporal state.
