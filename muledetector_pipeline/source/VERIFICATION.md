# Verification of the clean repository

The source is the verified shared implementation from
[`aml_data-engineering2`, commit `0bcb7ff`](https://github.com/RN90210/aml_data-engineering2/commit/0bcb7ff24fc129fdedcec2033acb0d7458b5d29e).
The archive remains separate. This repository removes standalone feature runners,
old materialization applications, and the pipeline-selection branch. Necessary
helper functions were moved into `events.py`, `audit.py`, `outputs.py`, and
`connector.py`; the independent recomputation functions live only in `validation/`.

## Checks performed

- The four native state processors, their calculation helper, and the complete
  static feature builder match the verified source after line-ending normalization.
- Fifteen runtime/preparation definitions, including `SharedBatchRunner`, match
  their source text. `validation/source_checksums.json` records the checksums.
- Internal imports and module names passed to subprocess commands resolve inside
  this repository. The processing code cannot silently fall back to archived
  runner modules.
- A real clean `aml_replay` Airflow workflow processed 1,000 transactions in two
  batches of 500: 2,000 valid Account events, zero rejections, two ready packages,
  and a passed final report. It also exercised normal delivery with independent
  recomputation disabled.
- All twelve output tables were compared with the saved verified run using the
  same input and static metadata. Column order, column types, complete key sets,
  absence of duplicate/null keys, and numeric values within absolute `1e-6`
  tolerance matched.
- The Airflow image built from this repository's Dockerfile, OS packages, and
  pinned Python requirements. Tests run in a fresh container with only this
  repository mounted, including the opt-in Spark lookup-plan test.
- The Kafka Compose configuration renders successfully.

The clean workflow test reused existing prepared data and the existing local
Kafka/Airflow services. The fresh-image tests separately check dependency and
import isolation. They do not constitute a new deployment/load benchmark or a
claim that every machine has identical performance.

Recovery and rejected-batch behavior were verified on the source implementation;
the same coordinator and processors are retained here, with their control-unit
tests. This cleanup did not repeat the full 5,000-transaction replay or every
failure scenario. The clean 1,000-transaction output comparison verifies the
refactoring without changing the feature definitions.

## Development checks

Inside the scheduler shell, from `/opt/muledetector`:

```text
python -m pytest -q
```

The Spark plan test is opt-in because it starts Java/Spark:

```text
MULEDETECTOR_SPARK_PLAN_TEST=1 python -m pytest -q
```

The complete suite contains 41 checks, including the Spark integration test.
Without that flag, the integration test is skipped.

For feature-code changes, enable `verify_features` in the same Airflow workflow.
The functions in `validation/reference.py` are the independent test oracle, not
a supported second way to generate model-update packages.

`validation.verify_recovery` is an opt-in fault-injection utility. Use fresh
topic, run ID, output, checkpoint, and report paths. It intentionally fails after
an Edge write and before its commit, resumes the partial batch, checks that ready
batches remain unchanged, and compares recovered values. Never point it at a
working run's checkpoint root. Tests do not remove production checkpoints.
