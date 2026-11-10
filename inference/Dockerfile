# Frozen-model inference + live network UI. Build context: ml_ly_latest/ (see inference/compose.yaml).
FROM python:3.11.5-slim

RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 pyarrow==24.0.0 scikit-learn==1.4.2

WORKDIR /app
COPY config.py ./
COPY stage2/ stage2/
COPY network/ network/
COPY inference/*.py inference/
COPY inference/model/ inference/model/
# Only the standard-library package feed is needed from the pipeline source.
COPY muledetector_pipeline/source/streaming/__init__.py pipeline_source/streaming/__init__.py
COPY muledetector_pipeline/source/streaming/operations/ pipeline_source/streaming/operations/

ENV PYTHONUNBUFFERED=1
