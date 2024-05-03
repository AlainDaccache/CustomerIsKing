FROM apache/airflow:2.6.0-python3.8
USER root
RUN apt-get update \
    # && apt-get install -y --no-install-recommends \
    # build-essential my-awesome-apt-dependency-to-add \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir --user \
    sqlalchemy \
    mlflow \
    boto3 \
    psycopg2-binary \
    apache-airflow-providers-mongo \
    apache-airflow-providers-amazon \
    apache-airflow[amazon]

ARG INGESTION_DEPENDENCY="postgres,mlflow,datalake-s3"

# RUN pip install --no-cache-dir --user "openmetadata-managed-apis==1.0.0.0" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.4.1/constraints-3.8.txt"
# RUN pip install --no-cache-dir --user "openmetadata-ingestion[${INGESTION_DEPENDENCY}]==1.0.0.0"

# COPY --chown=airflow:0 ./ingestion_dependency.sh /opt/airflow
# Provide Execute Permissions to shell script
# RUN chmod +x /opt/airflow/ingestion_dependency.sh

# Make required folders for openmetadata-airflow-apis
RUN mkdir -p /opt/airflow/dag_generated_configs

# COPY --chown=airflow:0 ingestion/examples/airflow/dags /opt/airflow/dags
COPY --chown=airflow:0 ./dags /opt/airflow/dags
COPY --chown=airflow:0 ./logs /opt/airflow/logs

# RUN chown airflow:0 /opt/airflow/dags
# RUN chown airflow:0 /opt/airflow/logs

RUN airflow db init && airflow users create \
    --username ${AIRFLOW_ADMIN_USER:-admin} \
    --firstname Alain \
    --lastname Daccache \
    --role Admin \
    --email spiderman@superhero.org \
    --password ${AIRFLOW_ADMIN_PASSWORD:-admin}