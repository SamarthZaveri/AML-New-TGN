"""The single Airflow workflow: prepare, publish, process, optionally verify, report."""
from __future__ import annotations
import csv
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from airflow.sdk import dag, task
from pendulum import datetime

PROJECT_ROOT = Path(os.environ.get("AML_PROJECT_ROOT_IN_CONTAINER", "/opt/muledetector"))
RAW_TRANSACTIONS_PATH = Path(os.environ.get("AML_TRANSACTIONS_PATH", PROJECT_ROOT / "data/raw/HI-Medium_Trans.csv"))
ACCOUNT_ENTITY_LOOKUP_PATH = Path(os.environ.get("AML_ACCOUNT_ENTITY_LOOKUP_PATH", PROJECT_ROOT / "data/processed/account_entity_lookup.parquet"))
STATIC_OUTPUT_DIR = Path(os.environ.get("AML_STATIC_OUTPUT_DIR", PROJECT_ROOT / "data/processed/spark_static_full"))
BOOTSTRAP_SERVERS = os.environ.get("AML_KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
REPLAY_START_TIMESTAMP = "2022/09/01 00:30"
REPLAY_END_TIMESTAMP = "2022/09/02 00:30"
REPLAY_ROW_COUNT = 1000
MAX_OFFSETS_PER_TRIGGER = 500


def run_project_command(task_name: str, command: list[str]) -> None:
    """Run one project command and send its output to the Airflow task log."""
    if not PROJECT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Mounted project directory not found: {PROJECT_ROOT}"
        )

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")

    print(f"\nStarting task: {task_name}")
    print(f"Command: {shlex.join(command)}\n")

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


@dag(dag_id="aml_replay", description="Shared Kafka/Spark model-update replay",
     start_date=datetime(2026, 9, 1), schedule=None, catchup=False,
     max_active_runs=1, max_active_tasks=1,
     default_args={"owner": "Ruhan", "retries": 0}, tags=["aml", "kafka", "spark"])
def aml_replay():
    @task
    def build_run_config(**context) -> dict[str, str]:
        """
        Create small run-specific settings for downstream tasks.

        This dictionary is safe for Airflow XCom because it contains only
        names and paths, never Parquet data or Spark DataFrames.
        """
        run_timestamp = context['dag_run'].run_after.strftime('%Y%m%dT%H%M%S')
        run_id = f'airflow_replay_{run_timestamp}'
        settings = context['dag_run'].conf or {}
        allowed = {
            'replay_row_count', 'max_offsets_per_trigger', 'prepared_replay_path',
            'concurrent_feature_jobs', 'verify_features',
        }
        if set(settings) - allowed:
            raise ValueError('Unsupported run settings: ' + ', '.join(sorted(set(settings) - allowed)))
        replay_row_count = int(settings.get('replay_row_count', REPLAY_ROW_COUNT))
        max_offsets = int(settings.get('max_offsets_per_trigger', MAX_OFFSETS_PER_TRIGGER))
        if replay_row_count <= 0 or max_offsets <= 0:
            raise ValueError('Replay row count and max offsets must be positive.')
        feature_jobs = int(settings.get('concurrent_feature_jobs', 2))
        if not 1 <= feature_jobs <= 5:
            raise ValueError('concurrent_feature_jobs must be from 1 to 5.')
        verify_features = settings.get('verify_features', True)
        if not isinstance(verify_features, bool):
            raise ValueError('verify_features must be a JSON boolean.')
        streaming_root = PROJECT_ROOT / 'data/streaming'
        checkpoint_root = PROJECT_ROOT / 'data/checkpoints/airflow'
        return {
            'run_id': run_id,
            'concurrent_feature_jobs': str(feature_jobs),
            'verify_features': 'true' if verify_features else 'false',
            'replay_row_count': str(replay_row_count),
            'max_offsets_per_trigger': str(max_offsets),
            'topic': f'aml-transactions-{run_timestamp}',
            'replay_path': str(streaming_root / 'replay_windows' / f'{run_id}.csv'),
            'prepared_replay_path': settings.get('prepared_replay_path', ''),
            'model_updates_dir': str(streaming_root / 'model_updates'),
            'checkpoint_root': str(checkpoint_root / run_id),
            'audit_rejected_dir': str(streaming_root / 'audit/rejected_account_events'),
            'audit_metrics_dir': str(streaming_root / 'audit/metrics'),
            'report_path': str(streaming_root / 'reports' / f'{run_id}.json'),
        }

    @task
    def prepare_replay_window(config: dict[str, str]) -> None:
        """Create one ordered, bounded replay CSV from the mounted raw file."""
        if config["prepared_replay_path"]:
            source = Path(config["prepared_replay_path"])
            destination = Path(config["replay_path"])
            if not source.is_file():
                raise FileNotFoundError(
                    f"Prepared replay CSV not found: {source}"
                )
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = csv.reader(handle)
                next(rows, None)
                row_count = sum(1 for row in rows if row)
            if row_count != int(config["replay_row_count"]):
                raise ValueError(
                    f"Prepared replay has {row_count} rows; expected "
                    f"{config['replay_row_count']}."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            return
        run_project_command(
            "prepare_replay_window",
            [
                "python",
                "-m",
                "streaming.operations.prepare_replay_window",
                "--input-path",
                str(RAW_TRANSACTIONS_PATH),
                "--start-timestamp",
                REPLAY_START_TIMESTAMP,
                "--end-timestamp",
                REPLAY_END_TIMESTAMP,
                "--max-rows",
                config["replay_row_count"],
                "--output-path",
                config["replay_path"],
            ],
        )

    @task
    def publish_replay_to_kafka(config: dict[str, str]) -> None:
        """Publish the prepared replay file to this run's unique Kafka topic."""
        run_project_command(
            "publish_replay_to_kafka",
            [
                "python",
                "-m",
                "streaming.kafka.producer",
                "--input-path",
                config["replay_path"],
                "--max-rows",
                config["replay_row_count"],
                "--delay-seconds",
                "0",
                "--topic",
                config["topic"],
                "--bootstrap-servers",
                BOOTSTRAP_SERVERS,
            ],
        )

    @task
    def process_updates(config: dict[str, str]) -> None:
        """One application prepares once and publishes each completed batch."""
        run_project_command("process_updates", [
            "python", "-m", "streaming.ingestion.pipeline",
            "--topic", config["topic"], "--bootstrap-servers", BOOTSTRAP_SERVERS,
            "--max-offsets-per-trigger", config["max_offsets_per_trigger"],
            "--run-id", config["run_id"],
            "--account-entity-lookup-path", str(ACCOUNT_ENTITY_LOOKUP_PATH),
            "--static-metadata-dir", str(STATIC_OUTPUT_DIR),
            "--output-dir", config["model_updates_dir"],
            "--rejected-output-dir", config["audit_rejected_dir"],
            "--metrics-output-dir", config["audit_metrics_dir"],
            "--checkpoint-dir", str(Path(config["checkpoint_root"]) / "shared"),
            "--concurrent-feature-jobs", config["concurrent_feature_jobs"],
        ])

    @task.branch(trigger_rule="none_failed_min_one_success")
    def choose_verification(config: dict[str, str]) -> str:
        """Retain recomputation for acceptance; normal delivery can omit it."""
        return "validate_replay" if config["verify_features"] == "true" else "write_run_report"

    @task
    def validate_replay(config: dict[str, str]) -> None:
        """Rebuild expected values from the replay CSV and compare outputs."""
        run_project_command(
            "validate_replay",
            [
                "python",
                "-m",
                "validation.validate_replay",
                "--replay-path",
                config["replay_path"],
                "--account-entity-lookup-path",
                str(ACCOUNT_ENTITY_LOOKUP_PATH),
                "--static-output-dir",
                str(STATIC_OUTPUT_DIR),
                "--model-updates-output-dir",
                config["model_updates_dir"],
                "--run-id",
                config["run_id"],
                "--max-offsets-per-trigger",
                config["max_offsets_per_trigger"],
            ],
        )

    @task(trigger_rule="none_failed_min_one_success")
    def write_run_report(config: dict[str, str]) -> None:
        """Write one final JSON report only when the replay has no rejections."""
        run_project_command(
            "write_run_report",
            [
                "python",
                "-m",
                "streaming.operations.run_report",
                "--run-id",
                config["run_id"],
                "--metrics-output-dir",
                config["audit_metrics_dir"],
                "--model-updates-output-dir",
                config["model_updates_dir"],
                "--report-path",
                config["report_path"],
                "--require-no-rejections",
            ],
        )

    config = build_run_config()
    prepared = prepare_replay_window(config)
    published = publish_replay_to_kafka(config)
    processed = process_updates(config)
    verification = choose_verification(config)
    validated = validate_replay(config)
    reported = write_run_report(config)
    prepared >> published >> processed >> verification
    verification >> validated >> reported
    verification >> reported


aml_replay()
