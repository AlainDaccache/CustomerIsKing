from constants import *
import argparse
from data_handler import FileReader, ChromaCRUD, EmbeddingLoader
from model_handler import LLMLoader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import time


def launch_terminal(qa):

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        start_time = time.time()
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]
        end_time = time.time()
        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        print("Time it took (in sec):", end_time - start_time)
        source_docs = [doc.metadata["source"] for doc in res["source_documents"]]
        print("Source documents:\n", source_docs)
        # print(res)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.path.join(ROOT_DIRECTORY, ".env"))
    print(os.environ["HF_HOME"])
    parser = argparse.ArgumentParser(
        description="Command-line tool for LLM and data ingestion"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Command to spin up LLM
    llm_parser = subparsers.add_parser("spin_up_llm_from_local", help="Spin up an LLM")
    llm_parser.add_argument(
        "--model_type",
        default=MODEL_TYPE,
        help=f"Type of the model (default: {MODEL_TYPE})."
        f"Currently support Llama.cpp, HuggingFace, or GPT4All",
    )
    llm_parser.add_argument(
        "--model_id", default=MODEL_ID, help=f"ID of the model (default: {MODEL_ID})"
    )
    llm_parser.add_argument(
        "--model_basename",
        default=MODEL_BASENAME,
        help=f"Optional basename of the model (default: {MODEL_BASENAME})",
    )
    llm_parser.add_argument(
        "--context_window_size",
        type=int,
        default=CONTEXT_WINDOW_SIZE,
        help=f"Window size for LLM (default: {CONTEXT_WINDOW_SIZE})",
    )
    llm_parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=N_GPU_LAYERS,
        help=f"Number of GPU layers (default: {N_GPU_LAYERS})",
    )
    llm_parser.add_argument(
        "--n_batch", type=int, default=N_BATCH, help=f"Batch size (default: {N_BATCH})"
    )
    llm_parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Maximum new tokens (default: {MAX_NEW_TOKENS})",
    )
    llm_parser.add_argument(
        "--cpu_percentage",
        type=int,
        default=CPU_PERCENTAGE,
        help=f"CPU percentage (default: {CPU_PERCENTAGE})",
    )
    llm_parser.add_argument(
        "--device_type",
        default=DEVICE_TYPE,
        help=f"Device type (default: {DEVICE_TYPE})",
    )
    llm_parser.add_argument(
        "--use_memory",
        default=USE_MEMORY,
        help=f"Use memory in chain (default: {USE_MEMORY})",
    )
    llm_parser.add_argument(
        "--system_prompt",
        default=SYSTEM_PROMPT,
        help=f"System prompt (default: {SYSTEM_PROMPT})",
    )
    llm_parser.add_argument(
        "--chain_type", default=CHAIN_TYPE, help=f"Chain type (default: {CHAIN_TYPE})"
    )
    llm_parser.add_argument(
        "--chroma_host", type=str, help="Chroma host address", default=CHROMA_DB_HOST
    )
    llm_parser.add_argument(
        "--chroma_port", type=int, help="Chroma port number", default=CHROMA_DB_PORT
    )
    llm_parser.add_argument(
        "--chroma_collection_name",
        type=str,
        help="Name of the collection",
        default=CHROMA_DB_COLLECTION,
    )
    llm_parser.add_argument(
        "--embedding_model_name",
        default=EMBEDDING_MODEL_NAME,
        help=f"Embedding model name (default: {EMBEDDING_MODEL_NAME})",
    )

    # # Command to ingest data
    # ingest_data_from_paths_parser = subparsers.add_parser(
    #     "ingest_data_from_paths", help="Ingest data"
    # )
    # ingest_data_from_paths_parser.add_argument(
    #     "paths", nargs="+", help="List of file paths or folder path"
    # )
    # ingest_data_from_paths_parser.add_argument(
    #     "--db_chunk_size",
    #     type=int,
    #     default=DB_CHUNK_SIZE,
    #     help=f"Database chunk size (default: {DB_CHUNK_SIZE})",
    # )
    # ingest_data_from_paths_parser.add_argument(
    #     "--db_chunk_overlap",
    #     type=int,
    #     default=DB_CHUNK_OVERLAP,
    #     help=f"Database chunk overlap (default: {DB_CHUNK_OVERLAP})",
    # )
    # ingest_data_from_paths_parser.add_argument(
    #     "--embedding_model_name",
    #     default=EMBEDDING_MODEL_NAME,
    #     help=f"Embedding model name (default: {EMBEDDING_MODEL_NAME})",
    # )

    ingest_data_from_mode_parser = subparsers.add_parser(
        "ingest_data_from_mode", help="Ingest data"
    )
    ingest_data_from_mode_parser.add_argument(
        "--full", action="store_true", help="Full mode (default)"
    )
    ingest_data_from_mode_parser.add_argument(
        "--incremental", action="store_true", help="Incremental mode"
    )
    ingest_data_from_mode_parser.add_argument(
        "--db_chunk_size",
        type=int,
        default=DB_CHUNK_SIZE,
        help=f"Database chunk size (default: {DB_CHUNK_SIZE})",
    )
    ingest_data_from_mode_parser.add_argument(
        "--db_chunk_overlap",
        type=int,
        default=DB_CHUNK_OVERLAP,
        help=f"Database chunk overlap (default: {DB_CHUNK_OVERLAP})",
    )
    ingest_data_from_mode_parser.add_argument(
        "--embedding_model_name",
        default=EMBEDDING_MODEL_NAME,
        help=f"Embedding model name (default: {EMBEDDING_MODEL_NAME})",
    )

    ingest_data_from_mode_parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        help="MLFlow tracking URI",
        default=os.getenv("MLFLOW_TRACKING_URI"),
    )
    ingest_data_from_mode_parser.add_argument(
        "--mlflow_registry_uri",
        type=str,
        help="MLFlow registry URI",
        default=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
    )
    ingest_data_from_mode_parser.add_argument(
        "--embedding_model_uri",
        type=str,
        help="URI of the embedding model",
        default=None,
    )
    ingest_data_from_mode_parser.add_argument(
        "--chroma_host", default=CHROMA_DB_HOST, type=str, help="Chroma host address"
    )
    ingest_data_from_mode_parser.add_argument(
        "--chroma_port", default=CHROMA_DB_PORT, type=int, help="Chroma port number"
    )
    ingest_data_from_mode_parser.add_argument(
        "--chroma_collection_name",
        default=CHROMA_DB_COLLECTION,
        type=str,
        help="Name of the collection",
    )
    ingest_data_from_mode_parser.add_argument(
        "--minio_access_key", help="Minio access key", default=None
    )
    ingest_data_from_mode_parser.add_argument(
        "--minio_secret_key", help="Minio secret key", default=None
    )
    ingest_data_from_mode_parser.add_argument(
        "--minio_bucket_name", help="Minio bucket name", default=None
    )
    ingest_data_from_mode_parser.add_argument(
        "--minio_object_prefix", help="Prefix for objects in Minio bucket", default=None
    )
    ingest_data_from_mode_parser.add_argument(
        "--embeddings_model_source", help="Embeddings Model Source", default="local"
    )
    ingest_data_from_mode_parser.add_argument(
        "--data_source", help="Data Source", default="local"
    )

    # Command to spin up LLM
    spin_up_llm_from_mlflow_parser = subparsers.add_parser(
        "spin_up_llm_from_mlflow", help="Spin up an LLM from MLFlow"
    )
    spin_up_llm_from_mlflow_parser.add_argument(
        "--mlflow_tracking_uri", type=str, help="MLFlow tracking URI"
    )
    spin_up_llm_from_mlflow_parser.add_argument(
        "--mlflow_registry_uri", type=str, help="MLFlow registry URI"
    )
    spin_up_llm_from_mlflow_parser.add_argument(
        "--llm_model_uri", type=str, help="URI of the LLM model in MLFlow"
    )
    spin_up_llm_from_mlflow_parser.add_argument(
        "--embedding_model_uri", type=str, help="URI of the embedding model"
    )
    spin_up_llm_from_mlflow_parser.add_argument(
        "--chroma_host", type=str, help="Chroma host address"
    )
    spin_up_llm_from_mlflow_parser.add_argument(
        "--chroma_port", type=int, help="Chroma port number"
    )
    spin_up_llm_from_mlflow_parser.add_argument(
        "--chroma_collection_name", type=str, help="Name of the collection"
    )

    # Command to register embedding model
    register_embedding_parser = subparsers.add_parser(
        "register_embedding_model", help="Register an embedding model"
    )
    register_embedding_parser.add_argument(
        "--hugging_face_repo_id",
        help="Hugging Face repository ID",
        default=EMBEDDING_MODEL_NAME,
    )
    register_embedding_parser.add_argument(
        "--hugging_face_filename", help="Hugging Face filename", default=None
    )

    # Command to register LLM
    register_llm_parser = subparsers.add_parser("register_llm", help="Register an LLM")
    register_llm_parser.add_argument(
        "--hugging_face_repo_id",
        help="Hugging Face repository ID",
        default=MODEL_ID,
    )
    register_llm_parser.add_argument(
        "--hugging_face_filename", help="Hugging Face filename", default=MODEL_BASENAME
    )

    args = parser.parse_args()

    if args.command == "spin_up_llm_from_mlflow":
        from model_handler import LLMLoader

        print("ARGUMENTS:")
        print(args)
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        qa = LLMLoader().load_from_mlflow(
            callback_manager=callback_manager,
            embedding_model_uri=args.embedding_model_uri,
            chroma_host=args.chroma_host,
            chroma_port=args.chroma_port,
            collection_name=args.chroma_collection_name,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_registry_uri=args.mlflow_registry_uri,
            llm_model_uri=args.llm_model_uri,
        )
        launch_terminal(qa=qa)

    if args.command == "spin_up_llm_from_local":
        print("ARGUMENTS:")
        print(args)
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        embedding_model = EmbeddingLoader().load_from_local(
            model_name=args.embedding_model_name
        )
        qa = LLMLoader().load_from_local(
            embedding_model=embedding_model,
            chroma_host=args.chroma_host,
            chroma_port=args.chroma_port,
            collection_name=args.chroma_collection_name,
            model_type=args.model_type,
            model_id=args.model_id,
            model_basename=args.model_basename,
            context_window_size=args.context_window_size,
            n_gpu_layers=args.n_gpu_layers,
            n_batch=args.n_batch,
            max_new_tokens=args.max_new_tokens,
            cpu_percentage=args.cpu_percentage,
            use_memory=args.use_memory,
            system_prompt=args.system_prompt,
            chain_type=args.chain_type,
            callback_manager=callback_manager,
        )
        print("QA:", qa)
        launch_terminal(qa=qa)

    elif args.command == "ingest_data_from_mode":
        print("ARGUMENTS:")
        print(args)
        if args.embeddings_model_source == "local":
            assert args.embedding_model_name is not None
            embedding_model = EmbeddingLoader().load_from_local(
                model_name=args.embedding_model_name
            )
        elif args.embeddings_model_source == "mlflow":
            assert args.embedding_model_uri is not None
            assert args.mlflow_registry_uri is not None
            assert args.mlflow_tracking_uri is not None
            embedding_model = EmbeddingLoader().load_from_mlflow(
                mlflow_tracking_uri=args.mlflow_tracking_uri,
                mlflow_registry_uri=args.mlflow_registry_uri,
                embedding_model_uri=args.embedding_model_uri,
            )

        vector_client = ChromaCRUD(
            embedding_model=embedding_model,
            host=args.chroma_host,
            port=args.chroma_port,
            collection=args.chroma_collection_name,
        )

        if not (args.full ^ args.incremental):
            parser.error("Please specify either --full or --incremental")

        mode = "full" if args.full else "incremental"

        collection_name = args.chroma_collection_name
        if mode == "full":
            vector_client.reset()

            # vector_client.client.get_or_create_collection(
            #     name=args.chroma_collection_name, embedding_function=embedding_model
            # )

        file_reader = FileReader(vector_client=vector_client)

        if args.data_source == "minio":
            # Handle ingesting data from Minio bucket
            from data_handler import MinioClient

            minio_host_port = args.mlflow_registry_uri.split("http://")[-1]
            print("Minio Host/Port:", minio_host_port)
            minio_client = MinioClient(
                endpoint_url=minio_host_port,
                access_key=args.minio_access_key,
                secret_key=args.minio_secret_key,
            )
            file_reader.ingest_from_blob(
                blob_client=minio_client,
                bucket_name=args.minio_bucket_name,
                download_path=os.path.join(DATA_FOLDER_PATH, args.minio_bucket_name),
            )
        elif args.data_source == "local":
            file_reader.ingest_data(mode="full")
        else:
            raise Exception()

    elif args.command == "register_embedding_model":
        print("ARGUMENTS:")
        print(args)
        from llm_embedding_pipeline import download_and_register_embeddings

        download_and_register_embeddings(
            hugging_face_repo_id=args.hugging_face_repo_id,
            hugging_face_filename=args.hugging_face_filename,
            mlflow_registry_uri=args.mlflow_registry_uri,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )
    elif args.command == "register_llm":
        print("ARGUMENTS:")
        print(args)
        from llm_embedding_pipeline import download_and_register_llm

        download_and_register_llm(
            hugging_face_repo_id=args.hugging_face_repo_id,
            hugging_face_filename=args.hugging_face_filename,
            mlflow_registry_uri=args.mlflow_registry_uri,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )
    else:
        print("Invalid command")
