# Marketing

# Overview

# Installation

Download file from: https://communityfoundations.ca/wp-content/uploads/2021/08/HR-Guide_-Policy-and-Procedure-Template.pdf. Move under ./fileserver/uploads.

As well as: https://archive.ics.uci.edu/dataset/352/online+retail. You'll have to Save As .csv, 
then go to a CSV to JSON converter

```

```

## Pre-Requisites

* Check you have git downloaded: `git --help`
* Check that you have **Python 3.7 or greater**: `python3 --version`
* Check that you have **Docker 20.10.0 or greater**: `docker --version`
* Check that you have **Docker Compose v2.1.1 or greater**: `docker compose version`

If on Windows, need WSL2. 
* minimum memory requirements: 12 GB
    * go  %UserProfile%\.wslconfig and ensure memory=12GB

# Guide

Refer to this table for URLs and authentication

|Microservice|Category         |Client URL            |User|Password|
|------------|-----------------|----------------------|----|--------|
|MongoDB     |NoSQL Database   |http://localhost:8081 |root|example|
|Minio       |Data Lake        |http://localhost:9001 |minio|qwerty98|
|ML Flow     |ML Ops           |http://localhost:5001 |mlflow|mlflow
|Postgres    |Data Warehouse   |http://localhost:5432 |alan|qwerty98| 
|Airflow     |Data Orchestrator|http://localhost:8080 |airflow|airflow |
|PyPi        |Package Manager  |http://localhost:5050 | pypi-user | qwerty98 |



## Environment Variables & Accesses

Then, `cd` to `customer-360-demo` and create an `.env` file by downloading, dragging and dropping the file from [here](https://drive.google.com/file/d/11WPtqkpgqf3Z0vkP3Pj145IvxWpPmb2R/view?usp=drive_link).

Then, in the same directory, do the following:

## Required Config before Docker Composing the Project 


### Postgres

In the terminal, do the same: `sudo chmod -R 777 postgres`, then, `docker-compose up -d postgres`.

Go on Docker Desktop, then click on the **postgres** container

![Postgres Docker Container](assets/postgres_container_docker.png)

Go on the "Terminal" then, do the following (will be helpful for the data observability part):

```
psql -h localhost -U alan # write qwerty98 for password when prompted
CREATE ROLE root WITH LOGIN SUPERUSER PASSWORD 'password';
exit
psql -U root postgres
CREATE EXTENSION pg_stat_statements;
alter system set shared_preload_libraries='pg_stat_statements';
```

Then restart the container.
 
### Airlfow
```
echo -e "AIRFLOW_UID=$(id -u)" >> .env

mkdir ./logs ./plugins
chmod -R 777 airflow/dags/
chmod -R 777 airflow/logs/
chmod -R 777 airflow/plugins/

docker-compose up airflow-init
```


### PyPI

Server that holds our Python packages for performing ML Modelling.

```
mkdir pypi-server && cd pypi-server && mkdir auth && cd auth
export PYPI_USER=pypi-user
pip install passlib
htpasswd -sc .htpasswd $PYPI_USER # you will be prompted for the password, type qwerty98
```

Now, go back to your own terminal, and `docker-compose up -d --build`.

## Setup for the rest of the project

### Minio

Go to [minio](http://localhost:9001) and login (`user: minio`, `password: qwerty98`). Create the following buckets:

|Bucket|
|------|
|`ml-bucket`|
|`my-ecom-transactions`|
|`my-knowledge-base`|

### MongoDB

Head to [localhost:8081](http://localhost:8081) and create a database called `my_ecom_mongodb`.

Then `python3 scripts/ftp_generate_data.py` on your terminal (if `python3` doesn't work, try `python`).
This will load the initial data into our source systems, namely the HTTP Server (Mock CRM) and Mongo DB.

### Airflow

Head to [localhost:8080](http://localhost:8080) and login (user and password are `admin`)
In the top bar, hover over Admin > Connections. Add the following connections:

|Connection ID | Connection Type | Host | Port |Schema| User | Password | Extra |
|--------------|-----------------|------|------|------|---------|------|-------|
|ecom_mongodb| MongoDB | mongo |27017 | <leave empty> | root | example ||
|ecom_postgres| Postgres | postgres | 5432 | postgres | alan | qwerty98||
|data-lake| Amazon Web Services | <leave empty> | <leave empty> | <leave empty> | <leave empty> | <leave empty> | {"aws_access_key_id": "minio",  "aws_secret_access_key": "qwerty98", "endpoint_url": "http://minio:9000",  "region_name": "us-east-1"}| 

### Relational SQL DB Client

Download `DBeaver` and create a database connection for Postgres. 
* Driver: Postgres
* Host: localhost
* Port: 5433
* Database: postgres
* Username: alan
* Password: qwerty98

### LLM

* Ubuntu 22.04
* CUDA: 12.3
* GPU: NVIDIA GeForce RTX 2070
* Nvidia-SMI 545.34
* Driver Version 546.26
* Python 3.10.6
Add nvidia container runtime to Docker:
https://medium.com/htc-research-engineering-blog/nvidia-docker-on-wsl2-f891dfe34ab
```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info|grep -i runtime
 Runtimes: nvidia runc
 Default Runtime: runc
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.3.1-base-ubuntu20.04 nvidia-smi
sudo docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.13.5/install-guide.html


If using WSL then on Docker Desktop -> Settings -> Docker Engine, 

```
{
    "runtimes":{
        "nvidia":{
            "path":"/usr/bin/nvidia-container-runtime",
            "runtimeArgs":[
                
            ]
        }
    }
}
```

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
python -m pip install openpyxl langchain-community gradio accelerate bitsandbytes kaleido python-multipart langchain chromadb langchainhub bs4 InstructorEmbedding sentence-transformers docx2txt gpt4all unstructured[pdf] pysqlite3 mlflow minio boto3
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```
sudo apt-get install git-lfs
git lfs install
sudo git clone https://huggingface.co/hkunlp/instructor-large
```

```
import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

```

**Running the LLM from terminal, using MLFlow as registry**
```
python -m terminal spin_up_llm_from_mlflow \
    --mlflow_tracking_uri http://mlflow:5001 \
    --mlflow_registry_uri http://minio:9000 \
    --embedding_model_uri "runs:/ebdbabdb63784baf9160a4d86ab2a371/model" \
    --llm_model_uri "runs:/ccbe9d062d0341a1a41d42fd4d3b2457/model" \
    --chroma_host chroma \
    --chroma_port 8000 \
    --collection_name operations-collection

```

**Using the SDK to run the LLM**
```python
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pysqlite3
import sys
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
from mlflow.pyfunc import load_model

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# print("Model URI:", model_info.model_uri)

llm_run_uri = "runs:/ccbe9d062d0341a1a41d42fd4d3b2457/model"
loaded_llm_mlflow = load_model(
            model_uri=llm_run_uri
        ).unwrap_python_model() # to retrieve our class
		
chroma_host = "chroma"
chroma_port = "8000"
collection_name = "operations-collection"
embedding_model_run_id = "ebdbabdb63784baf9160a4d86ab2a371"

embedding_model_uri = "runs:/{}/model".format(embedding_model_run_id)

qa_retriever = loaded_llm_mlflow.init_qa_bot(
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            embedding_model_uri=embedding_model_uri,
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            collection_name=collection_name,
        )
print(qa_retriever)
```
### Jupyter

Go [here](http://localhost:8888/lab/tree/work/customer_lifetime_value.ipynb) and run all the cells (it will stop at the "API" section at the end because
we haven't gotten to that point yet, it's normal).

### MLFlow

We run the experiment of the Airflow DAG [here](http://localhost:8080/dags/lifetime-value-model-training/grid).

Then head to a specific experiment result after running the Airflow DAG, by clicking on “clv_experiment_frequency” on the side, then on the link below “Created” in the resulting table i.e. the “25 minutes ago” in the picture below. 

![MLFlow showing an experiment run for the CLV models](assets/mlflow-experiment.png)

 As you can see under “Metrics”, you can track the performance of your models as you R&D over time. Click on “Register Model”, “Create New Model” and name it “Pareto_NBD” for the “clv_experiment_frequency”. Head back to experiments and do the same for the “clv_experiment_monetary”, naming the model “Gamma_Gamma_CLV”.

![MLFlow showing the configuration of the model](assets/mlflow-experiment-results.png)

You can go here http://localhost:5001/#/models/Pareto_NBD/versions/1 to observe the expected inputs and outputs of the model, optionally adding Tags and Descriptions so that end users know how to use this API. 

![MLFlow showing the configuration of the model](assets/mlflow-experiment-results.png)

You can stage that model to production. And do the same for the other model. Now, those models have a stable URI so that when you have new model versions, you can swap them without redeploying the API everytime. 

![MLFlow Staging -> Production](assets/mlflow-stage-transition.png)

Then head to ML Flow to see the experiments results, `http://localhost:5001`. You can observe the results of the experiment here http://localhost:5001/#/experiments/1 after running the data pipeline. You can track the performance of your models as you R&D over time.

You can see the model artifact which contains all the code and dependencies required to train the model. Notice `customer-360` as part of the dependencies.

MLFlow registers the model behind the scenes (as part of the **experiment**) in the model registry, being **Minio** in our case, http://localhost:9001/browser/ml-bucket.

You can deploy the model on MLFLow. You register each of them first; name the monetary one `Gamma_Gamma_CLV` and the frequency one `BG_NBD` (or `Pareto_NBD` depending which one performed better). Just make sure you update the docker command in the Flask API if the Pareto_NBD performed better (FREQUENCY_ML_MODEL_URI).

Then head over to models and deploy into production.

### Flask API

After deploying the model on MLFlow, you can associate it to an API so we can serve the predictions.

```
docker build -t api api

docker run -d -t -i --name api -p 5000:5000 \
-e MONETARY_ML_MODEL_URI="models:/Gamma_Gamma_CLV/Production" \
-e FREQUENCY_ML_MODEL_URI="models:/Pareto_NBD/Production" \
-e MLFLOW_TRACKING_URI="http://mlflow:5001" \
-e MLFLOW_S3_ENDPOINT_URL="http://minio:9000" \
-e AWS_ACCESS_KEY_ID="minio" \
-e AWS_SECRET_ACCESS_KEY="qwerty98" \
-e POSTGRES_HOST="postgres" \
-e POSTGRES_PORT="5432" \
-e POSTGRES_USER="alan" \
-e POSTGRES_PWD="qwerty98" \
--network=customer-360-demo_app_net \
api  
```

## Phase III: Data Observability

Tools Used:
* OpenMetadata
* Dash

### Dashboard


### OpenMetadata

Head over to `http://localhost:8585/` and sign in as "admin" for both username and password. Go to "Settings"


#### Postgres:
For Postgres, under "Services" on the side bar, click on "Databases" and "+ Add". Alternatively, head [here](http://localhost:8585/databaseServices/add-service), 
select Postgres, put any service name i.e. Postgres, and fill the following config. 
* Connection Schema: postgresql+psycopg2
* Username: alan
* Password: qwerty98
* Host and Port: host.docker.internal:5433
* Database: postgres

You can "Test the Connection" for sanity. Then click on save, then add ingestion (keep default settings). Head [here](http://localhost:8585/table/Postgres.postgres.public.dim_customers) to verify it's successful.

#### Airflow

For Airflow, under "Services", go on "Pipelines", add one and name it "Airflow".
* Host and Port: http://host.docker.internal:8080
* Metadata Database Connection: PostgresConnection
* PostgresConnection
    * Connection Schema: postgresql+psycopg2
    * Username: alan
    * Password: qwerty98
    * Host and Port: host.docker.internal:5433
    * Database: airflow

Test the connection, then add and trigger the ingestion (keep default settings).

#### MinIO

For MinIO, go to "Storages" under "Services", specify "S3", name it "MinIO", and fill the config:
* AWS Access Key ID: minio
* AWS Secret Access Key: qwerty98
* AWS Region: us-east-1
* Endpoint URL: http://minio:9000

when testing the connection on this one, the ListMetric failure is expected. OpenMetaData doesn't fully support MinIO yet, only S3.

Then, add and trigger the ingestion (keep default settings).

#### ML Flow

Go to the "ML Models" under "Services", select "ML Flow"
MLFlow:
* Tracking URI: http://mlflow:5001
* Registry URI: postgresql://alan:qwerty98@postgres:5432/mlflow

Then test the connection, save, add and trigger the ingestion (keep default settings).