import hashlib
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.mongo.hooks.mongo import MongoHook
from airflow.providers.amazon.aws.operators.s3 import S3Hook
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
from bson.json_util import dumps
from io import StringIO
from airflow.operators.docker_operator import DockerOperator
import docker
import requests
import os
import json
import pandas as pd
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
from airflow import DAG
from airflow.operators.empty import EmptyOperator

S3_CONN_ID = "minio"
S3_TRANSACTIONS_BUCKET = "my-ecom-transactions"
S3_KNOWLEDGE_BUCKET = "my-knowledge-base"
S3_TARGET_KEY = f"{datetime.today().strftime('%Y-%m-%d')}.json"

POSTGRES_CONN_ID = "ecom_postgres"
POSTGRES_DB_NAME = "my_ecom_sqldb"
POSTGRES_SCHEMA_NAME = "CUSTOMER_TRANSACTIONS"
POSTGRES_CUSTOMERS_DIM_NAME = "DIM_CUSTOMERS"
POSTGRES_ITEMS_DIM_NAME = "DIM_ITEMS"
POSTGRES_TRANSACTIONS_FACT_NAME = "FACT_TRANSACTIONS"


MONGO_DB_CONN_ID = "ecom_mongodb"
MONGODB_DB_NAME = "my_ecom_mongodb"
MONGO_DB_COLLECTION_NAME = "customer-transactions"

###########################     HELPER FUNCTIONS        ########################


def s3_to_transact_df(bucket_name):
    s3_hook = S3Hook(S3_CONN_ID)
    local_file_path = s3_hook.download_file(key=S3_TARGET_KEY, bucket_name=bucket_name)
    with open(local_file_path, "r") as f:
        j = json.load(f)

    df = pd.DataFrame(j)
    df.drop(columns=["_id"], inplace=True)
    df = df.applymap(lambda x: np.nan if isinstance(x, dict) else x)
    df = df[~df["CustomerID"].isna()]
    return df, local_file_path


# def s3_to_customer_df(bucket_name):
#     s3_hook = S3Hook(S3_CONN_ID)
#     local_file_path = s3_hook.download_file(key=S3_TARGET_KEY, bucket_name=bucket_name)
#     df = pd.read_csv(local_file_path)
#     # df.drop(columns=["_id"], inplace=True)
#     df = df[~df["CustomerID"].isna()]
#     return df, local_file_path


#################################################################################

# from pathlib import Path


def download_files_to_local():

    base_url = "http://fileserver:8080"
    FILENAMES = requests.get(f"{base_url}/files").json()["files"]
    # FILENAMES = [Path(f).stem for f in fn_response]
    FILE_DOWNLOADS_URLS = [f"{base_url}/download/{f}" for f in FILENAMES]

    for url, filename in zip(FILE_DOWNLOADS_URLS, FILENAMES):
        try:
            response = requests.get(url)
            # Save the downloaded file locally
            with open(filename, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"An error occurred: {e}")
    return FILENAMES


def upload_files_to_minio(bucket_name, filenames):
    s3_hook = S3Hook(aws_conn_id=S3_CONN_ID)
    for filename in filenames:
        s3_hook.load_file(
            filename=filename,
            key=filename,
            bucket_name=bucket_name,
            replace=True,
        )


def el_pdfs_from_http_to_s3(last_loaded_transaction_date: datetime):
    FILENAMES = download_files_to_local()
    print("FILENAMES:", FILENAMES)
    upload_files_to_minio(bucket_name=S3_KNOWLEDGE_BUCKET, filenames=FILENAMES)


def el_raw_transactions_mongodb_to_s3(last_loaded_transaction_date: datetime):
    register_adapter(np.int64, AsIs)
    mongodb_hook = MongoHook(conn_id=MONGO_DB_CONN_ID)
    mongodb_client = mongodb_hook.get_conn()
    print(f"Connected to MongoDB - {mongodb_client.server_info()}")
    mongodb_db = mongodb_client.get_database(name=MONGODB_DB_NAME)
    mongo_db_transactions_coll = mongodb_db.get_collection(
        name=MONGO_DB_COLLECTION_NAME
    )

    # filter only above last loaded transaction_date
    transaction_cursor = mongo_db_transactions_coll.find()
    json_data = dumps(list(transaction_cursor))
    # print(json_data)

    with open("json_temp.json", "w") as f:
        f.write(json_data)

    s3_hook = S3Hook(S3_CONN_ID)
    with open("json_temp.json", "rb") as f:
        s3_hook.load_file_obj(
            file_obj=f,
            bucket_name=S3_TRANSACTIONS_BUCKET,
            key=S3_TARGET_KEY,
            replace=True,
        )

    os.remove("json_temp.json")


def etl_customers_s3_to_postgres_dim():
    customers_df, local_file_path = s3_to_transact_df(
        bucket_name=S3_TRANSACTIONS_BUCKET
    )

    customers_df = (
        customers_df.rename({"CustomerID": "ID"}, axis=1)
        .drop_duplicates(subset=["ID"])
        .reset_index(drop=True)
    )

    customers_df["ID"] = customers_df["ID"].astype(int).astype(str)
    print("ID is unique?", customers_df["ID"].is_unique)
    postgres_sql_upload = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    postgres_sql_upload.insert_rows(
        schema=POSTGRES_SCHEMA_NAME,
        table=POSTGRES_CUSTOMERS_DIM_NAME,
        rows=customers_df[["ID", "Country"]].values,
    )


def etl_items_s3_to_postgres_dim():
    items_df, local_file_path = s3_to_transact_df(bucket_name=S3_TRANSACTIONS_BUCKET)
    print("Items DF:", items_df.head())
    relevant_columns = ["StockCode", "Description", "UnitPrice", "InvoiceDate"]
    items_df = items_df[relevant_columns].rename(
        {
            "StockCode": "STOCK_CODE",
            "Description": "DESCRIPTION",
            "UnitPrice": "UNIT_PRICE",
            "InvoiceDate": "INVOICE_DATE",
        },
        axis=1,
    )
    items_df["DESCRIPTION"] = items_df.groupby("STOCK_CODE")["DESCRIPTION"].transform(
        lambda x: x.ffill()
    )
    items_df["INVOICE_DATE"] = pd.to_datetime(items_df["INVOICE_DATE"])
    cols = ["STOCK_CODE", "DESCRIPTION", "UNIT_PRICE"]
    items_df["ID"] = items_df[cols].apply(
        lambda row: "|".join(row.values.astype(str)), axis=1
    )
    items_df["ID"] = items_df["ID"].apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()
    )
    items_df = items_df[
        ["ID", "STOCK_CODE", "DESCRIPTION", "UNIT_PRICE", "INVOICE_DATE"]
    ]
    print("Items df before groupby:\n", items_df.head())
    items_df = (
        items_df.groupby(["ID", "STOCK_CODE", "UNIT_PRICE", "DESCRIPTION"])[
            "INVOICE_DATE"
        ]
        .min()
        .reset_index()
    )
    items_df = items_df.sort_values(["STOCK_CODE", "INVOICE_DATE"])
    items_df["DATETIME_VALID_TO"] = items_df.groupby("STOCK_CODE", group_keys=False)[
        "INVOICE_DATE"
    ].apply(lambda x: x.shift(-1).fillna(datetime(2099, 12, 31)))
    items_df["DATETIME_VALID_TO"] = items_df["DATETIME_VALID_TO"] - timedelta(seconds=1)
    items_df.rename({"INVOICE_DATE": "DATETIME_VALID_FROM"}, axis=1, inplace=True)
    postgres_sql_upload = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    postgres_sql_upload.insert_rows(
        schema=POSTGRES_SCHEMA_NAME, table=POSTGRES_ITEMS_DIM_NAME, rows=items_df.values
    )
    os.remove(local_file_path)


def etl_transactions_s3_to_postgres_fact():
    transactions_df, local_file_path = s3_to_transact_df(
        bucket_name=S3_TRANSACTIONS_BUCKET
    )
    relevant_columns = [
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "StockCode",
        "Description",
        "UnitPrice",
        "Quantity",
    ]
    transactions_df = transactions_df[relevant_columns].rename(
        {
            "CustomerID": "CUSTOMER_ID",
            "InvoiceNo": "INVOICE_NO",
            "StockCode": "STOCK_CODE",
            "Description": "DESCRIPTION",
            "UnitPrice": "UNIT_PRICE",
            "InvoiceDate": "INVOICE_DATE",
            "Quantity": "QUANTITY",
        },
        axis=1,
    )
    transactions_df["QUANTITY"] = transactions_df["QUANTITY"].fillna(1)
    transactions_df = (
        transactions_df.drop_duplicates()
        .groupby(
            ["CUSTOMER_ID", "INVOICE_NO", "STOCK_CODE", "UNIT_PRICE", "DESCRIPTION"]
        )
        .agg({"INVOICE_DATE": "max", "QUANTITY": "sum"})
        .reset_index()
    )
    cols = ["STOCK_CODE", "DESCRIPTION", "UNIT_PRICE"]
    transactions_df["ITEM_ID"] = transactions_df[cols].apply(
        lambda row: "|".join(row.values.astype(str)), axis=1
    )
    transactions_df["ITEM_ID"] = transactions_df["ITEM_ID"].apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()
    )
    transactions_df.drop(cols, axis=1, inplace=True)
    print("Ensure Invoice_No and Item_Id represent the row ID")
    print(
        transactions_df.shape[0]
        == transactions_df[["INVOICE_NO", "ITEM_ID"]].drop_duplicates().shape[0]
    )
    transactions_df = transactions_df[
        ["CUSTOMER_ID", "ITEM_ID", "INVOICE_NO", "INVOICE_DATE", "QUANTITY"]
    ]
    transactions_df["CUSTOMER_ID"] = (
        transactions_df["CUSTOMER_ID"].astype(float).astype(int).astype(str)
    )
    postgres_sql_upload = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    postgres_sql_upload.insert_rows(
        schema=POSTGRES_SCHEMA_NAME,
        table=POSTGRES_TRANSACTIONS_FACT_NAME,
        rows=transactions_df.values,
    )
    os.remove(local_file_path)


with DAG(
    dag_id="elt_ftp_mongodb_customers_to_postgres",
    schedule_interval=None,
    start_date=datetime.today(),
    catchup=False,
    tags=["ecomm", "customer", "transactions"],
    default_args={"owner": "Alain", "retries": 2},
    params={
        "embedding_model_uri": "runs:/20e9a0caf94a454aa775e4db84825f63/model",
        "minio_bucket_name": "my-knowledge-base",
        "chroma_collection_name": "operations-collection",
        "docker_image_name": "llm",
    },
    user_defined_macros={"json": json},
) as dag:
    dummy_start_operator = EmptyOperator(task_id="start", dag=dag)

    dummy_end_operator = EmptyOperator(task_id="end", dag=dag)

    create_schema = PostgresOperator(
        task_id="create_schema",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=f"CREATE SCHEMA IF NOT EXISTS {POSTGRES_SCHEMA_NAME};",
    )

    get_last_loaded_transaction_date = 0  # TODO
    el_raw_transactions_mongodb_to_s3_task = PythonOperator(
        python_callable=el_raw_transactions_mongodb_to_s3,
        task_id="el_raw_transactions_mongodb_to_s3",
        dag=dag,
        op_kwargs={"last_loaded_transaction_date": get_last_loaded_transaction_date},
    )

    el_pdfs_from_http_to_s3_task = PythonOperator(
        python_callable=el_pdfs_from_http_to_s3,
        task_id="el_pdfs_from_http_to_s3",
        dag=dag,
        op_kwargs={"last_loaded_transaction_date": get_last_loaded_transaction_date},
    )

    ingest_embeddings_from_s3_to_chromadb_task = DockerOperator(
        task_id="ingest_embeddings_from_s3_to_chromadb_task",
        image="{{ params.docker_image_name }}",
        command=[
            "sh",
            "-c",
            """
        cd src && \

        python -m terminal ingest_data_from_mode \
        --mlflow_tracking_uri http://{{ var.value.mlflow_host }}:{{ var.value.mlflow_port }} \
        --mlflow_registry_uri {{ json.loads(conn.minio.extra)['endpoint_url'] }} \
        --embedding_model_uri "{{ params.embedding_model_uri }}"  \
        --chroma_host {{ var.value.chroma_host }} \
        --chroma_port {{ var.value.chroma_port }} \
        --chroma_collection_name "{{ params.chroma_collection_name }}" \
        --embeddings_model_source mlflow \
        --minio_access_key {{ json.loads(conn.minio.extra)['aws_access_key_id'] }} \
        --minio_secret_key {{ json.loads(conn.minio.extra)['aws_secret_access_key'] }} \
        --minio_bucket_name "{{ params.minio_bucket_name }}" \
        --data_source minio \
        --full

    """,
        ],
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        dag=dag,
        # FIXME for some reason param not substituting during runtime, hardcoding for now
        network_mode="customerisking_app_net",  # "{{ params.docker_network_name }}",  # "container:llm",
    )
    #######################################################################

    create_transactions_table = PostgresOperator(
        task_id="create_if_not_exists_transactions_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=f"""
        CREATE TABLE IF NOT EXISTS {POSTGRES_TRANSACTIONS_FACT_NAME} (
                CUSTOMER_ID     VARCHAR                 ,
                ITEM_ID         VARCHAR                 ,
                INVOICE_NO      VARCHAR                 ,
                INVOICE_DATE    TIMESTAMP               ,
                QUANTITY        INTEGER                 ,
                PRIMARY KEY (INVOICE_NO, ITEM_ID)       ,
                CONSTRAINT fk_customer
                    FOREIGN KEY(CUSTOMER_ID)
                        REFERENCES {POSTGRES_CUSTOMERS_DIM_NAME}(ID),
                CONSTRAINT fk_item
                    FOREIGN KEY(ITEM_ID)
                        REFERENCES {POSTGRES_ITEMS_DIM_NAME}(ID)
        );
        """,
    )

    task_etl_transactions_s3_to_postgres_fact = PythonOperator(
        task_id="etl_transactions_s3_to_postgres_fact",
        python_callable=etl_transactions_s3_to_postgres_fact,
        dag=dag,
    )
    ############################################################

    create_customers_table = PostgresOperator(
        task_id="create_if_not_exists_customers_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=f"""
        CREATE TABLE IF NOT EXISTS {POSTGRES_CUSTOMERS_DIM_NAME} (
                ID                  VARCHAR,
                COUNTRY             VARCHAR,
                PRIMARY KEY         (ID)
        );
        """,
    )
    truncate_customers_table = PostgresOperator(
        task_id="truncate_customers_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=f"TRUNCATE TABLE {POSTGRES_CUSTOMERS_DIM_NAME} CASCADE;",
    )

    task_etl_customers_s3_to_postgres_dim = PythonOperator(
        task_id="etl_customers_s3_to_postgres_dim",
        python_callable=etl_customers_s3_to_postgres_dim,
        dag=dag,
    )

    ############################################################

    create_items_table = PostgresOperator(
        task_id="create_if_not_exists_items_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=f"""
        CREATE TABLE IF NOT EXISTS {POSTGRES_ITEMS_DIM_NAME} (
                ID                      VARCHAR                 ,
                STOCK_CODE              VARCHAR                 ,
                UNIT_PRICE              DECIMAL                 ,
                DESCRIPTION             VARCHAR                 ,
                DATETIME_VALID_FROM     TIMESTAMP               ,
                DATETIME_VALID_TO       TIMESTAMP               ,
                PRIMARY KEY (ID)
        );
        """,
    )
    truncate_items_table = PostgresOperator(
        task_id="truncate_items_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=f"TRUNCATE TABLE {POSTGRES_ITEMS_DIM_NAME} CASCADE;",
    )

    task_etl_items_s3_to_postgres_dim = PythonOperator(
        task_id="etl_items_s3_to_postgres_dim",
        python_callable=etl_items_s3_to_postgres_dim,
        dag=dag,
    )

    #########################################################################################

    (
        dummy_start_operator
        >> el_raw_transactions_mongodb_to_s3_task
        >> create_schema
        >> [create_customers_table, create_items_table]
        >> create_transactions_table
        >> [truncate_items_table, truncate_customers_table]
    )

    (
        dummy_start_operator
        >> el_pdfs_from_http_to_s3_task
        >> ingest_embeddings_from_s3_to_chromadb_task
        >> dummy_end_operator
    )
    truncate_customers_table >> task_etl_customers_s3_to_postgres_dim
    truncate_items_table >> task_etl_items_s3_to_postgres_dim
    task_etl_items_s3_to_postgres_dim >> task_etl_transactions_s3_to_postgres_fact
    task_etl_customers_s3_to_postgres_dim >> task_etl_transactions_s3_to_postgres_fact
    task_etl_transactions_s3_to_postgres_fact >> dummy_end_operator
