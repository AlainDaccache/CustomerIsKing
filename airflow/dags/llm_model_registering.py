from airflow import DAG
from datetime import timedelta
from airflow.operators.docker_operator import DockerOperator
from airflow.utils.dates import days_ago
import docker

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "register_models_mlflow",
    default_args=default_args,
    params={
        "embedding_hf_repo_id": "hkunlp/instructor-large",
        "embedding_hf_filename": None,
        "llm_hf_repo_id": "TheBloke/Llama-2-7b-Chat-GGUF",
        "llm_hf_filename": "llama-2-7b-chat.Q4_K_M.gguf",
        "docker_image_name": "llm",
        "docker_network_name": "customerisking_app_net",  # for network stack,
    },
    description="A DAG to register embedding and LLM models into MLFlow",
    schedule_interval=None,
    start_date=days_ago(1),
    tags=["mlflow", "models"],
)

# Install required packages and register embedding model task
register_embedding_model = DockerOperator(
    task_id="register_embedding_model",
    image="{{ params.docker_image_name }}",
    # command=""" python -m pip install minio mlflow boto3 && python -m pip install -r requirements.txt && python -m terminal register_embedding_model --hugging_face_repo_id "{{ params.embedding_hf_repo_id }}" {% if params.embedding_hf_filename %}--hugging_face_filename "{{ params.embedding_hf_filename }}" {% endif %}""",
    command=[
        "sh",
        "-c",
        """cd src && python -m terminal register_embedding_model --hugging_face_repo_id "{{ params.embedding_hf_repo_id }}" """,
    ],
    dag=dag,
    # need to rebuild airflow image for that and pip install docker
    device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
    # FIXME for some reason param not substituting during runtime, hardcoding for now
    network_mode="container:llm",  # "{{ params.docker_network_name }}"
)

# Install required packages and register LLM model task
register_llm_model = DockerOperator(
    task_id="register_llm_model",
    image="{{ params.docker_image_name }}",
    command=[
        "sh",
        "-c",
        """
        cd src && \
        python -m terminal register_llm \
        --hugging_face_repo_id "{{ params.llm_hf_repo_id }}" \
        --hugging_face_filename "{{ params.llm_hf_filename }}"
    """,
    ],
    device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
    dag=dag,
    # FIXME for some reason param not substituting during runtime, hardcoding for now
    network_mode="container:llm",  # "{{ params.docker_network_name }}"
)

# Define task dependencies
register_embedding_model >> register_llm_model
