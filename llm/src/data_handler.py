import os
import time

from dotenv import load_dotenv
from minio.error import InvalidResponseError
from minio import Minio
from langchain.embeddings import HuggingFaceInstructEmbeddings
import pysqlite3
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from chromadb import HttpClient
from chromadb.config import Settings
from constants import (
    LAST_INGESTION_FILE,
    DATA_FOLDER_PATH,
    DB_CHUNK_OVERLAP,
    DB_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    TOP_K_SEARCH_ARGS,
    CACHE_PATH,
    CHROMA_DB_COLLECTION,
    CHROMA_DB_HOST,
    CHROMA_DB_PORT,
    DEVICE_TYPE,
    MODELS_PATH,
    ROOT_DIRECTORY,
)
import os
import locale
from langchain.document_loaders import (
    CSVLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv(dotenv_path=os.path.join(ROOT_DIRECTORY, ".env"))


class MinioClient:

    def __init__(self, endpoint_url, access_key, secret_key):
        print("HERE!!!!")
        print(endpoint_url, access_key, secret_key)
        self.minio_client = self._get_minio_client(endpoint_url, access_key, secret_key)

    def _get_minio_client(self, endpoint_url, access_key, secret_key):
        # download all files from bucket

        # Todo enable SSL
        minio_client = Minio(
            endpoint_url, access_key=access_key, secret_key=secret_key, secure=False
        )
        return minio_client

    def download_files_from_bucket(self, bucket_name, download_path):
        try:
            # Get list of objects in the bucket
            objects = self.minio_client.list_objects(bucket_name, recursive=True)
            for obj in objects:
                # Download each object
                self.minio_client.fget_object(
                    bucket_name, obj.object_name, f"{download_path}/{obj.object_name}"
                )
                print(f"Downloaded {obj.object_name}")
        except InvalidResponseError as err:
            print(err)


class FileReader:

    def __init__(self, vector_client) -> None:
        self.last_ingestion_file = LAST_INGESTION_FILE
        self.data_folder_path = DATA_FOLDER_PATH
        self.db_chunk_size = DB_CHUNK_SIZE
        self.db_chunk_overlap = DB_CHUNK_OVERLAP
        self.embedding_model_name = EMBEDDING_MODEL_NAME
        self.vector_client = vector_client

    def read_last_ingestion_time(self):
        if os.path.exists(self.last_ingestion_file):
            with open(self.last_ingestion_file, "r") as file:
                last_time = file.read().strip()
                return int(last_time)
        return None

    def update_last_ingestion_time(self):
        current_time = str(int(time.time()))  # Convert current time to timestamp
        with open(self.last_ingestion_file, "w") as file:
            file.write(current_time)

    def get_new_files(self, last_time):
        if last_time is None:
            last_time = 0  # minimum time
        print("Last time", time.ctime(last_time))

        new_files = []
        for root, dirs, files in os.walk(self.data_folder_path):
            for file in files:
                filepath = os.path.join(root, file)
                # we get the max because not all files recently moved have a
                # recent modified time. so in that case, we get the created time
                # this behaviour happens during drag/drop, copy/paste file the modified
                # time stays what it was, but the created time changes to NOW
                modified_time = max(
                    os.path.getmtime(filepath), os.path.getctime(filepath)
                )
                print("File:", filepath, "Modified_time", time.ctime(modified_time))
                if modified_time > float(last_time):
                    new_files.append(filepath)
        return new_files

    def get_paths_from_folder(self, folder):
        file_paths = []

        for root, dirs, files in os.walk(folder):
            for f in files:
                file_path = os.path.join(root, f)
                file_paths.append(file_path)

            for d in dirs:
                subfolder = os.path.join(root, d)
                file_paths.extend(self.get_paths_from_folder(subfolder))

        return file_paths

    def ingest(self, file_paths):
        DOCUMENT_MAP = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".py": TextLoader,
            # ".pdf": PDFMinerLoader,
            ".pdf": UnstructuredFileLoader,
            ".csv": CSVLoader,  # UnstructuredCSVLoader,  # CSVLoader,
            # ".json": JSONLoader,
            ".xls": UnstructuredExcelLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
        }

        locale.getpreferredencoding = lambda: "UTF-8"
        text_documents = []
        # file_paths = preprocess(file_paths)
        for file_path in file_paths:
            print("Processing file", file_path)
            file_extension = os.path.splitext(file_path)[1]

            if file_extension not in DOCUMENT_MAP:
                print(f"File extension {file_extension} not supported currently")
                continue

            loader_class = DOCUMENT_MAP[file_extension]
            if file_extension == ".txt":
                loader = loader_class(file_path, encoding="UTF-8")
            else:
                loader = loader_class(file_path)

            try:
                text_documents.extend(loader.load())
            except Exception as e:
                print(f"Exception with file '{file_path}': {str(e)}")
                continue

        # FIXME current bug on chunk overlap greater than size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=256
        )
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=DB_CHUNK_SIZE,
        #     chunk_overlap=DB_CHUNK_OVERLAP,
        # )
        texts = text_splitter.split_documents(text_documents)

        print("Loader Class:", loader_class)
        print("Text Documents:", text_documents)
        print("Texts:", texts)
        self.vector_client.add_documents(texts=texts)

    def ingest_data(
        self,
        paths=None,
        mode=None,
    ):
        if paths is None and mode is None:
            raise Exception("Bad use")

        if mode == "incremental":
            last_ingestion_time = self.read_last_ingestion_time()
            file_paths = self.get_new_files(last_ingestion_time)
            print(
                f"Incremental mode. Ingesting new files since last ingestion: {file_paths}"
            )
            self.update_last_ingestion_time()
        elif mode == "full":
            # TODO remove all existing embeddings from VectorDB
            print("Full mode. Ingesting all files from the 'data' folder.")
            print("Data Folder Path:", self.data_folder_path)

            file_paths = self.get_paths_from_folder(folder=self.data_folder_path)
            self.update_last_ingestion_time()
        elif mode is not None:
            raise Exception("Invalid ingestion mode")
        else:
            pass
        if paths is not None:
            file_paths = []
            for path in paths:
                print(path)
                # abs_path = os.path.join(self.data_folder_path, path)
                if os.path.isdir(path):
                    file_paths += self.get_paths_from_folder(folder=path)
                else:
                    file_paths.append(path)
        unique_file_paths = list(set(file_paths))
        print("File paths:\n", unique_file_paths)
        if len(file_paths):
            self.ingest(file_paths=unique_file_paths)

    # TODO abstract away, make oop design e.g. for incremental load interface
    def ingest_from_blob(
        self, blob_client: MinioClient, bucket_name: str, download_path: str
    ):
        print("bucket_name:", bucket_name)
        print("download_path", download_path)
        blob_client.download_files_from_bucket(
            bucket_name=bucket_name, download_path=download_path
        )
        # files = []
        # for root, dirs, f in os.walk(download_path):
        #     for file in f:
        #         files.append(os.path.join(root, file))
        # todo allow incremental
        self.ingest_data(paths=[download_path])


def is_present_in_cache(model_name, cache_path):
    import re

    model_name_fmt = model_name.replace("/", "--")
    print("Model name fmt:", model_name_fmt)
    folders_under_local_hub = [i for i in os.listdir(cache_path)]
    for f in folders_under_local_hub:
        print("Folder trial:", f)
        if re.search(pattern=model_name_fmt, string=f):
            print("Match Found!")
            model_snapshot_path = os.path.join(cache_path, f, "snapshots")
            print("Model Snapshot Path:", model_snapshot_path)
            first_snapshot = os.listdir(model_snapshot_path)[0]
            print("First snapshot:", first_snapshot)
            exact_match = os.path.join(model_snapshot_path, first_snapshot)
            print("Exact match:", exact_match)
            return True, exact_match
    return False, None


class EmbeddingLoader:
    def __init__(self, from_local=True):
        self.from_local = from_local

    def load_from_local(self, model_name, device_type=DEVICE_TYPE):
        print("Model name:", model_name, device_type)
        print("Home of HF:", os.environ["HF_HOME"])

        print("Cache path:", CACHE_PATH)
        is_match, exact_match = is_present_in_cache(
            model_name=model_name, cache_path=CACHE_PATH
        )
        if is_match:
            model_path = exact_match
        else:
            from huggingface_hub import hf_hub_download, snapshot_download

            model_path = snapshot_download(
                repo_id=model_name,
                resume_download=True,
                cache_dir=CACHE_PATH,
            )
        print("Before feeding:", model_path)
        # more of a path
        embedding = HuggingFaceInstructEmbeddings(
            model_name=model_path, model_kwargs={"device": device_type}
        )
        return embedding

    def load_from_mlflow(
        self,
        mlflow_tracking_uri,
        mlflow_registry_uri,
        embedding_model_uri,
        device_type=DEVICE_TYPE,
    ):
        # TODO
        import mlflow
        import sys
        import pysqlite3

        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_registry_uri(mlflow_registry_uri)
        embedding_model_mlflow = mlflow.pyfunc.load_model(
            embedding_model_uri
        ).unwrap_python_model()
        embedding_model_lambda = embedding_model_mlflow.model_lambda
        embedding_model = embedding_model_lambda(device_type=device_type)
        return embedding_model


class ChromaCRUD:
    def __init__(
        self,
        embedding_model: HuggingFaceInstructEmbeddings = None,
        host=CHROMA_DB_HOST,
        port=CHROMA_DB_PORT,
        collection=CHROMA_DB_COLLECTION,
        device_type=DEVICE_TYPE,
    ):
        if embedding_model is None:
            print(
                f"Embedding Model fed to ChromaCRUD is None. Defaulting to {EMBEDDING_MODEL_NAME}"
            )
            embedding_model = HuggingFaceInstructEmbeddings(
                model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type}
            )
        self.client = self._get_chroma_client(host=host, port=port)
        self.collection_name = collection
        print("Collection name:", self.collection_name)
        self.embedding_function = embedding_model
        self.chroma = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

        print(
            "Client Collection Inits:", self.client.get_collection(self.collection_name)
        )
        print("Mongo Collection Inits:", self.chroma._collection)

    def reset(self):
        self.chroma.delete_collection()
        self.chroma = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def _get_chroma_client(self, host, port):
        return HttpClient(
            host=host,
            port=port,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                is_persistent=True,
            ),
        )

    def add_documents(self, texts):
        print("Client Collections:", self.client.get_collection(self.collection_name))
        self.chroma.add_documents(documents=texts)

    def update_document(self, document_id, updated_text):
        self.chroma.update_document(document_id=document_id, updated_text=updated_text)

    def delete_document(self, document_id):
        self.chroma.delete_document(document_id=document_id)

    def morph(self, into="retriever", **kwargs):
        if into == "retriever":
            return self.chroma.as_retriever(
                search_kwargs={"k": kwargs.get("top_k", TOP_K_SEARCH_ARGS)}
            )
        else:
            raise ValueError(f"`into` {into} is not supported yet")

    def search(self, query_text, top_k=TOP_K_SEARCH_ARGS):
        return self.chroma.as_retriever(search_kwargs={"k": top_k}).search(query_text)
