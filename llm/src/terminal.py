from constants import *
import argparse
import time

import argparse
import os
import torch
from constants import *
from data_handler import ingest, load_db
from model_handler import spin_up_llm
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

LAST_INGESTION_FILE = "last_ingestion_time.txt"


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


def read_last_ingestion_time():
    if os.path.exists(LAST_INGESTION_FILE):
        with open(LAST_INGESTION_FILE, "r") as file:
            last_time = file.read().strip()
            return int(last_time)
    return None


def update_last_ingestion_time():
    current_time = str(int(time.time()))  # Convert current time to timestamp
    with open(LAST_INGESTION_FILE, "w") as file:
        file.write(current_time)


def get_new_files(last_time):
    if last_time is None:
        last_time = 0  # minimum time
    print("Last time", time.ctime(last_time))

    new_files = []
    for root, dirs, files in os.walk(DATA_FOLDER_PATH):
        for file in files:
            filepath = os.path.join(root, file)
            # we get the max because not all files recently moved have a
            # recent modified time. so in that case, we get the created time
            # this behaviour happens during drag/drop, copy/paste file the modified
            # time stays what it was, but the created time changes to NOW
            modified_time = max(os.path.getmtime(filepath), os.path.getctime(filepath))
            print("File:", filepath, "Modified_time", time.ctime(modified_time))
            if modified_time > float(last_time):
                new_files.append(filepath)
    return new_files


def get_paths_from_folder(folder):
    file_paths = []

    for root, dirs, files in os.walk(folder):
        for f in files:
            file_path = os.path.join(root, f)
            file_paths.append(file_path)

        for d in dirs:
            subfolder = os.path.join(root, d)
            file_paths.extend(get_paths_from_folder(subfolder))

    return file_paths


def ingest_data(
    paths=None,
    mode=None,
    db_chunk_size=DB_CHUNK_SIZE,
    db_chunk_overlap=DB_CHUNK_OVERLAP,
    embedding_model_name=EMBEDDING_MODEL_NAME,
):
    if paths is None and mode is None:
        raise Exception("Bad use")

    if mode == "incremental":
        last_ingestion_time = read_last_ingestion_time()
        file_paths = get_new_files(last_ingestion_time)
        print(
            f"Incremental mode. Ingesting new files since last ingestion: {file_paths}"
        )
        update_last_ingestion_time()
    elif mode == "full":
        # TODO remove all existing embeddings from VectorDB
        print("Full mode. Ingesting all files from the 'data' folder.")
        print("Data Folder Path:", DATA_FOLDER_PATH)

        file_paths = get_paths_from_folder(folder=DATA_FOLDER_PATH)
        update_last_ingestion_time()
    elif mode is not None:
        raise Exception("Invalid ingestion mode")
    else:
        pass

    if paths is not None:
        file_paths = []
        for path in paths:
            print(path)
            abs_path = os.path.join(DATA_FOLDER_PATH, path)
            if os.path.isdir(path):
                file_paths += get_paths_from_folder(folder=abs_path)
            else:
                file_paths.append(abs_path)

    print("File paths:\n", file_paths)
    if len(file_paths):
        ingest(
            file_paths=file_paths,
            chunk_size=db_chunk_size,
            chunk_overlap=db_chunk_overlap,
            embedding_model_name=embedding_model_name,
        )


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

    args = parser.parse_args()

    if args.command == "spin_up_llm":
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
        ingest_data(
            paths=args.paths,
            db_chunk_size=args.db_chunk_size,
            db_chunk_overlap=args.db_chunk_overlap,
            embedding_model_name=args.embedding_model_name,
        )

    elif args.command == "ingest_data_from_mode":
        print(args.full)
        print(args.incremental)
        if not (args.full ^ args.incremental):
            parser.error("Please specify either --full or --incremental")

        mode = "full" if args.full else "incremental"
        ingest_data(
            mode=mode,
            db_chunk_size=args.db_chunk_size,
            db_chunk_overlap=args.db_chunk_overlap,
            embedding_model_name=args.embedding_model_name,
        )
    else:
        print("Invalid command")
