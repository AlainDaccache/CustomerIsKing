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

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y libgl1 && \
    apt-get install -y poppler-utils && \
    apt-get install -y libcairo2-dev pkg-config python3-dev && \
    apt-get install -y libtesseract-dev && \
    apt-get install -y tesseract-ocr

# apt install -y libpython3.10-dev &&\ 
# apt install -y python3.10-distutils && \
WORKDIR /app

COPY . .

# Install required packages and create conda environment
RUN conda install -y python=3.10 && \
    conda install -y mlflow boto3 && \
    mlflow artifacts download --run-id ${LLM_RUN_ID:-9a35279ef3314dd09042c5836775a0e9} --artifact-path model/conda.yaml --dst-path . && \
    mlflow artifacts download --run-id ${LLM_RUN_ID:-9a35279ef3314dd09042c5836775a0e9} --artifact-path model/code/requirements.txt --dst-path . && \
    mv /app/model/conda.yaml /app/ && mv /app/model/code/requirements.txt /app/ && \
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" conda env create -f conda.yaml && conda clean -afy

# Set default conda environment
RUN echo "conda activate ${CONDA_ENV_NAME:-mlflow-env}" >> ~/.bashrc


