# Runnig the Frontend by loading the models from Mlflow and running on the same server

# Use a smaller base image
FROM continuumio/miniconda3 AS builder

ARG CONDA_ENV_NAME
ARG LLM_RUN_ID

# Set environment variables
ENV AWS_S3_ENDPOINT_URL=minio:9000
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=qwerty98
ENV AWS_DEFAULT_REGION=""
ENV MLFLOW_DB_NAME=mlflow
ENV MLFLOW_DB_USER=mlflow
ENV MLFLOW_ARTIFACT_BUCKET_NAME=ml-bucket
ENV MLFLOW_ARTIFACT_DIR_NAME=artifacts
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV MLFLOW_TRACKING_URI=http://mlflow:5001

RUN echo ${LLM_RUN_ID:-b7370963}

WORKDIR /app

COPY . .

# Install required packages and create conda environment
RUN conda install -y python=3.10 && \
    conda install -y mlflow && \
    mlflow artifacts download --run-id ${LLM_RUN_ID:-b7370963c55b4a70afcba1829e211bec} --dst-path . && \
    conda env create -f /app/model/conda.yaml && \
    conda clean -afy

# Set default conda environment
RUN echo "conda activate ${CONDA_ENV_NAME:-mlflow-env}" >> ~/.bashrc


