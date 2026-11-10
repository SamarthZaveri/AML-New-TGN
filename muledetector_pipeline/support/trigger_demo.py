"""Trigger the unchanged published DAG using the included prepared replay."""
import json
import subprocess
from pathlib import Path
conf = json.loads(Path('/opt/handover/demo.json').read_text())
subprocess.run(['airflow', 'dags', 'unpause', 'aml_replay'], check=True)
subprocess.run(['airflow', 'dags', 'trigger', 'aml_replay', '--conf', json.dumps(conf)], check=True)
