ARG BASE_IMAGE=astrocrpublic.azurecr.io/runtime:3.3-7
FROM ${BASE_IMAGE}
ARG DEPENDENCIES_PREINSTALLED=false
USER root
COPY source/orchestration/airflow/requirements.txt /tmp/pipeline-requirements.txt
RUN if [ "$DEPENDENCIES_PREINSTALLED" != "true" ]; then apt-get update && apt-get install -y --no-install-recommends openjdk-21-jre && rm -rf /var/lib/apt/lists/* && uv pip install --system -r /tmp/pipeline-requirements.txt; fi
COPY --chown=astro:0 source/ /opt/muledetector/
COPY --chown=astro:0 support/ /opt/handover/
RUN mkdir -p /opt/muledetector/data /opt/spark-ivy && chown -R astro:0 /opt/spark-ivy
ENV PYTHONPATH=/opt/muledetector
ENV SPARK_SUBMIT_OPTS="-Divy.home=/opt/spark-ivy -Divy.default.ivy.user.dir=/opt/spark-ivy"
ENV AML_PROJECT_ROOT_IN_CONTAINER=/opt/muledetector
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/muledetector/orchestration/airflow/dags
USER astro
WORKDIR /opt/muledetector
RUN python /opt/handover/check_runtime.py && python /opt/handover/warm_connector.py
LABEL org.opencontainers.image.source="https://github.com/RN90210/aml_pipeline" org.opencontainers.image.revision="78437bc2782579ef07bd1f00c6269281d82f647a" org.opencontainers.image.title="MuleDetector published pipeline handover"
