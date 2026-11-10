"""Verify the exact pipeline dependencies without installing anything."""
import importlib.metadata as m
from pathlib import Path
for line in Path('/opt/muledetector/orchestration/airflow/requirements.txt').read_text().splitlines():
    if '==' in line:
        name, expected = line.split('==')
        actual = m.version(name)
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, found {actual}')
print('Published dependency versions verified.')
