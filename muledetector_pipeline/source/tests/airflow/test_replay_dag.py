"""The repository exposes one processing workflow and no implementation switch."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("airflow")
from airflow.models import DagBag
from airflow.providers.standard.operators.python import BranchPythonOperator

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def dag(monkeypatch):
    monkeypatch.setenv("AML_PROJECT_ROOT_IN_CONTAINER", str(ROOT))
    bag = DagBag(dag_folder=str(ROOT / "orchestration/airflow/dags"), include_examples=False)
    assert not bag.import_errors, bag.import_errors
    return bag.dags["aml_replay"]


def config_for(dag, **settings):
    return dag.get_task("build_run_config").python_callable(
        dag_run=SimpleNamespace(run_after=datetime(2026, 11, 2, tzinfo=timezone.utc), conf=settings))


def test_one_workflow_and_one_processing_task(dag):
    assert set(dag.task_ids) == {"build_run_config", "prepare_replay_window", "publish_replay_to_kafka",
                                  "process_updates", "choose_verification", "validate_replay", "write_run_report"}
    assert dag.max_active_runs == dag.max_active_tasks == 1
    assert dag.get_task("publish_replay_to_kafka").downstream_task_ids == {"process_updates"}
    assert dag.get_task("process_updates").downstream_task_ids == {"choose_verification"}


def test_default_batch_contract(dag):
    config = config_for(dag)
    assert config["replay_row_count"] == "1000"
    assert config["max_offsets_per_trigger"] == "500"
    assert config["concurrent_feature_jobs"] == "2"
    assert config["verify_features"] == "true"
    assert "pipeline_mode" not in config


def test_process_command_uses_only_the_canonical_pipeline(dag, monkeypatch):
    commands = []
    task = dag.get_task("process_updates")
    monkeypatch.setitem(task.python_callable.__globals__, "run_project_command",
                        lambda name, command: commands.append(command))
    task.python_callable(config_for(dag))
    command = commands[0]
    assert command[command.index("-m") + 1] == "streaming.ingestion.pipeline"
    assert command[command.index("--checkpoint-dir") + 1].endswith("/shared")
    assert command[command.index("--concurrent-feature-jobs") + 1] == "2"


def test_optional_verification_always_preserves_reporting(dag):
    choose = dag.get_task("choose_verification")
    assert isinstance(choose, BranchPythonOperator)
    assert choose.python_callable(config_for(dag)) == "validate_replay"
    assert choose.python_callable(config_for(dag, verify_features=False)) == "write_run_report"
    assert dag.get_task("write_run_report").trigger_rule == "none_failed_min_one_success"


def test_unknown_options_are_rejected(dag):
    with pytest.raises(ValueError, match="Unsupported run settings"):
        config_for(dag, unknown_setting=True)
