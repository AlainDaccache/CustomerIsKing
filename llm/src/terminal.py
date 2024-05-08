from constants import *
import argparse
from data_handler import FileReader, ChromaCRUD
from model_handler import spin_up_llm
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import time


def launch_terminal(
    model_id=MODEL_ID,
    model_basename=MODEL_BASENAME,
    context_window_size=CONTEXT_WINDOW_SIZE,
    n_gpu_layers=N_GPU_LAYERS,
    n_batch=N_BATCH,
    max_new_tokens=MAX_NEW_TOKENS,
    cpu_percentage=CPU_PERCENTAGE,
    device_type=DEVICE_TYPE,
    use_memory=USE_MEMORY,
    system_prompt=SYSTEM_PROMPT,
    model_type=MODEL_TYPE,
    chain_type=CHAIN_TYPE,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    qa = spin_up_llm(
        model_type=model_type,
        model_id=model_id,
        model_basename=model_basename,
        context_window_size=context_window_size,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        max_new_tokens=max_new_tokens,
        cpu_percentage=cpu_percentage,
        device_type=device_type,
        use_memory=use_memory,
        system_prompt=system_prompt,
        chain_type=chain_type,
        callback_manager=callback_manager,
    )
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
    parser = argparse.ArgumentParser(
        description="Command-line tool for LLM and data ingestion"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Command to spin up LLM
    llm_parser = subparsers.add_parser("spin_up_llm", help="Spin up an LLM")
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

    # Command to ingest data
    ingest_data_from_paths_parser = subparsers.add_parser(
        "ingest_data_from_paths", help="Ingest data"
    )
    ingest_data_from_paths_parser.add_argument(
        "paths", nargs="+", help="List of file paths or folder path"
    )
    ingest_data_from_paths_parser.add_argument(
        "--db_chunk_size",
        type=int,
        default=DB_CHUNK_SIZE,
        help=f"Database chunk size (default: {DB_CHUNK_SIZE})",
    )
    ingest_data_from_paths_parser.add_argument(
        "--db_chunk_overlap",
        type=int,
        default=DB_CHUNK_OVERLAP,
        help=f"Database chunk overlap (default: {DB_CHUNK_OVERLAP})",
    )
    ingest_data_from_paths_parser.add_argument(
        "--embedding_model_name",
        default=EMBEDDING_MODEL_NAME,
        help=f"Embedding model name (default: {EMBEDDING_MODEL_NAME})",
    )

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

    ingest_data_from_minio_parser = subparsers.add_parser(
        "ingest_data_from_minio", help="Ingest data from Minio bucket"
    )
    # Add arguments for ingesting data from Minio bucket
    ingest_data_from_minio_parser.add_argument(
        "--endpoint_url", help="Minio endpoint URL"
    )
    ingest_data_from_minio_parser.add_argument("--access_key", help="Minio access key")
    ingest_data_from_minio_parser.add_argument("--secret_key", help="Minio secret key")
    ingest_data_from_minio_parser.add_argument(
        "--bucket_name", help="Minio bucket name"
    )
    ingest_data_from_minio_parser.add_argument(
        "--object_prefix", help="Prefix for objects in Minio bucket"
    )

    args = parser.parse_args()

    if args.command == "spin_up_llm":
        if args.from_mlflow:
            import mlflow

            run_id = args.run_id
            model_path_from_artifact = args.model_path
            model = mlflow.pyfunc.load_model(
                f"runs:/{run_id}/{model_path_from_artifact}"
            )

        launch_terminal(
            model_type=args.model_type,
            model_id=args.model_id,
            model_basename=args.model_basename,
            context_window_size=args.context_window_size,
            n_gpu_layers=args.n_gpu_layers,
            n_batch=args.n_batch,
            max_new_tokens=args.max_new_tokens,
            cpu_percentage=args.cpu_percentage,
            device_type=args.device_type,
            use_memory=args.use_memory,
            system_prompt=args.system_prompt,
            chain_type=args.chain_type,
        )
    elif args.command == "ingest_data_from_paths":
        vector_client = ChromaCRUD()
        file_reader = FileReader(vector_client=vector_client)
        file_reader.ingest_data(paths=args.paths)

    elif args.command == "ingest_data_from_mode":
        vector_client = ChromaCRUD()
        file_reader = FileReader(vector_client=vector_client)

        if not (args.full ^ args.incremental):
            parser.error("Please specify either --full or --incremental")

        mode = "full" if args.full else "incremental"

        collection_name = args.collection_name
        if mode == "full":
            if not vector_client.collection_exists(collection_name):
                # If collection doesn't exist, add it
                vector_client.create_collection(collection_name)
            else:
                vector_client.delete_collection(collection_name)
        file_reader.ingest_data(mode="full")

    elif args.command == "ingest_data_from_minio":
        # Handle ingesting data from Minio bucket
        vector_client = ChromaCRUD()
        file_reader = FileReader(vector_client=vector_client)
        from data_handler import MinioClient

        minio_client = MinioClient(
            endpoint_url=args.endpoint_url,
            access_key=args.access_key,
            secret_key=args.secret_key,
        )
        file_reader.ingest_from_blob(
            blob_client=minio_client,
            bucket_name=args.bucket_name,
            download_path=os.path.join(DATA_FOLDER_PATH, args.bucket_name),
        )
    elif args.command == "register_embedding_model":
        mlflow_client = None
        embedding_model_name = args.model_name
        device_type = args.device_type
    else:
        print("Invalid command")
