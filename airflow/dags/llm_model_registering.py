from airflow import DAG
from datetime import timedelta
from airflow.operators.docker_operator import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "embedding_hf_repo_id": "hkunlp/instructor-large",
    "embedding_hf_filename": None,
    "llm_hf_repo_id": "TheBloke/Llama-2-7b-Chat-GGUF",
    "llm_hf_filename": "llama-2-7b-chat.Q4_K_M.gguf",
}

dag = DAG(
    "register_models_mlflow",
    default_args=default_args,
    description="A DAG to register embedding and LLM models into MLFlow",
    schedule_interval=None,
    start_date=days_ago(1),
    tags=["mlflow", "models"],
)

# Install required packages and register embedding model task
register_embedding_model = DockerOperator(
    task_id="register_embedding_model",
    image="llm",
    command="""
    python -m pip install minio mlflow boto3 && \
    python -m terminal register_embedding_model \
        --hugging_face_repo_id "{{ dag_run.conf.embedding_hf_repo_id }}" \
        {% if dag_run.conf.embedding_hf_filename %}--hugging_face_filename "{{ dag_run.conf.embedding_hf_filename }}" {% endif %}
    """,
    dag=dag,
    network_mode="bridge",  # Adjust network mode as per your Docker setup
)

# Install required packages and register LLM model task
register_llm_model = DockerOperator(
    task_id="register_llm_model",
    image="llm",
    command="""
    python -m pip install minio mlflow boto3 && \
    python -m terminal register_llm \
        --hugging_face_repo_id "{{ dag_run.conf.llm_hf_repo_id }}" \
        --hugging_face_filename "{{ dag_run.conf.llm_hf_filename }}"
    """,
    dag=dag,
    network_mode="bridge",  # Adjust network mode as per your Docker setup
)

# Define task dependencies
register_embedding_model >> register_llm_model
