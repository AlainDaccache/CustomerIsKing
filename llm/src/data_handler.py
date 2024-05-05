import pysqlite3
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb

import os
import locale
import logging
import pandas as pd
from langchain.document_loaders import (
    CSVLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    JSONLoader,
    UnstructuredCSVLoader,
)
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings

from constants import (
    DATA_FOLDER_PATH,
    MODELS_PATH,
    EMBEDDING_MODEL_NAME,
    DEVICE_TYPE,
    PERSIST_DIRECTORY,
    DB_CHUNK_SIZE,
    DB_CHUNK_OVERLAP,
)
import re


"""
Incremental Load:

1. Delete: Check files from knowledge base that were deleted. To do so, check 
existing files in ChromaDB. If the file doesn't have a match with the knowledge
base, it has been deleted. So we delete it from the database
2. Add: Conversely, if a file is in the knowledge base but not in the DB, 
then we ingest that one.
3. Update: Finally, if the file has been modified since the last ingestion,
then we replace the existing document in the ChromaDB with the more recent one.
"""


def delete_doc_from_db(db, source):
    coll = db.get()  # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])

    ids_to_del = []

    for idx in range(len(coll["ids"])):

        id = coll["ids"][idx]
        metadata = coll["metadatas"][idx]

        if re.search(source, metadata["source"]).group():
            ids_to_del.append(id)

    db._collection.delete(ids_to_del)


def load_db(texts=None, embedding_model_name: str = EMBEDDING_MODEL_NAME):
    # embedding_model_path = os.path.join(MODELS_PATH, embedding_model_name.replace("/", "-"))
    # print("Embedding Model Path:", embedding_model_path)

    # if not os.path.exists(embedding_model_path):
    #     model_name = embedding_model_name
    # else:
    #     model_name = embedding_model_path
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": DEVICE_TYPE},
    )

    if texts is None:  # retrieve mode
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
            ),
        ).as_retriever()  # search_kwargs={"k": 4}
    else:  # upload mode
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
            ),
        )
        db.persist()
        return


# def preprocess(file_paths):
#     file_paths_output = []
#     for file_path in file_paths:
#         file_extension = os.path.splitext(file_path)[1]
#         if file_extension == ".csv":
#             data = pd.read_csv(file_path)
#             json_data = data.to_json(orient='records')
#             # Save the JSON data to a file
#             file_path_stripped = os.path.splitext(file_path)[0]
#             output_file_path = f'{file_path_stripped}.json'
#             file_paths_output.append(output_file_path)
#             with open(output_file_path, 'w') as output_file:
#                 output_file.write(json_data)
#         else:
#             file_paths_output.append(file_path)
#     return file_paths_output


def ingest(
    file_paths,
    chunk_size: int = DB_CHUNK_SIZE,
    chunk_overlap: int = DB_CHUNK_OVERLAP,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
):
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(text_documents)
    from constants import ROOT_DIRECTORY

    print(ROOT_DIRECTORY)
    print("Loader Class:", loader_class)
    print("Text Documents:", text_documents)
    print("Texts:", texts)
    print("Embedding Model Name:", embedding_model_name)
    load_db(texts=texts, embedding_model_name=embedding_model_name)
