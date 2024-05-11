# import gradio as gr
# from typing import Any
# from queue import Queue, Empty
# from langchain.callbacks.base import BaseCallbackHandler
# from threading import Thread
# from constants import *
# from langchain.callbacks.manager import CallbackManager
# from model_handler import LLMLoader
# from data_handler import EmbeddingLoader


# class QueueCallback(BaseCallbackHandler):
#     """Callback handler for streaming LLM responses to a queue."""

#     def __init__(self, q):
#         self.q = q

#     def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
#         self.q.put(token)

#     def on_llm_end(self, *args, **kwargs: Any) -> None:
#         return self.q.empty()


# def launch_app(qa):
#     q = Queue()
#     job_done = object()

#     def answer(question):
#         def task():
#             res = qa(question)
#             answer, docs = res["result"], res["source_documents"]
#             s = "\nHere are the relevant sources for this information:\n"
#             unique_sources = list(set([doc.metadata["source"] for doc in docs]))

#             for i, doc in enumerate(unique_sources):
#                 s += f"{i + 1}. {doc}\n"

#             q.put(s)
#             q.put(job_done)

#         t = Thread(target=task)
#         t.start()

#     s = """
#     Welcome to Dunya, the intelligent Chatbot for our Fictional Comany!
#     I am here to assist and guide you through various tasks. Currently, I excel in:
#       • Retrieving and synthesizing information from our extensive knowledge base, covering processes, regulations, and more. Feel free to ask questions such as your rights and obligations for an employee or contractor.

#     In the near future, I will also be capable of extracting data from our database, enhancing my capabilities even further.
#     """

#     with gr.Blocks() as demo:
#         chatbot = gr.Chatbot(
#             [[None, s]],
#         )
#         msg = gr.Textbox()
#         clear = gr.Button("Clear")

#         def user(user_message, history):
#             return "", history + [[user_message, None]]

#         def bot(history):
#             question = history[-1][0]
#             print("Question: ", question)
#             history[-1][1] = ""
#             answer(question=question)
#             while True:
#                 try:
#                     next_token = q.get(True, timeout=1)
#                     if next_token is job_done:
#                         break
#                     history[-1][1] += next_token
#                     yield history
#                 except Empty:
#                     continue

#         msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
#             bot, chatbot, chatbot
#         )
#         clear.click(lambda: None, None, chatbot, queue=False)

#     demo.queue()
#     demo.launch(
#         share=False,
#         debug=False,
#         server_name="0.0.0.0",
#         server_port=7860,
#         ssl_verify=False,
#     )


# if __name__ == "__main__":
#     from constants import *
#     from dotenv import load_dotenv
#     import argparse

#     load_dotenv(dotenv_path=os.path.join(ROOT_DIRECTORY, ".env"))
#     print(os.environ["HF_HOME"])
#     parser = argparse.ArgumentParser(
#         description="Command-line tool for LLM and data ingestion"
#     )
#     subparsers = parser.add_subparsers(dest="command")

#     # Command to spin up LLM
#     llm_parser = subparsers.add_parser("spin_up_llm_from_local", help="Spin up an LLM")
#     llm_parser.add_argument(
#         "--model_type",
#         default=MODEL_TYPE,
#         help=f"Type of the model (default: {MODEL_TYPE})."
#         f"Currently support Llama.cpp, HuggingFace, or GPT4All",
#     )
#     llm_parser.add_argument(
#         "--model_id", default=MODEL_ID, help=f"ID of the model (default: {MODEL_ID})"
#     )
#     llm_parser.add_argument(
#         "--model_basename",
#         default=MODEL_BASENAME,
#         help=f"Optional basename of the model (default: {MODEL_BASENAME})",
#     )
#     llm_parser.add_argument(
#         "--context_window_size",
#         type=int,
#         default=CONTEXT_WINDOW_SIZE,
#         help=f"Window size for LLM (default: {CONTEXT_WINDOW_SIZE})",
#     )
#     llm_parser.add_argument(
#         "--n_gpu_layers",
#         type=int,
#         default=N_GPU_LAYERS,
#         help=f"Number of GPU layers (default: {N_GPU_LAYERS})",
#     )
#     llm_parser.add_argument(
#         "--n_batch", type=int, default=N_BATCH, help=f"Batch size (default: {N_BATCH})"
#     )
#     llm_parser.add_argument(
#         "--max_new_tokens",
#         type=int,
#         default=MAX_NEW_TOKENS,
#         help=f"Maximum new tokens (default: {MAX_NEW_TOKENS})",
#     )
#     llm_parser.add_argument(
#         "--cpu_percentage",
#         type=int,
#         default=CPU_PERCENTAGE,
#         help=f"CPU percentage (default: {CPU_PERCENTAGE})",
#     )
#     llm_parser.add_argument(
#         "--device_type",
#         default=DEVICE_TYPE,
#         help=f"Device type (default: {DEVICE_TYPE})",
#     )
#     llm_parser.add_argument(
#         "--use_memory",
#         default=USE_MEMORY,
#         help=f"Use memory in chain (default: {USE_MEMORY})",
#     )
#     llm_parser.add_argument(
#         "--system_prompt",
#         default=SYSTEM_PROMPT,
#         help=f"System prompt (default: {SYSTEM_PROMPT})",
#     )
#     llm_parser.add_argument(
#         "--chain_type", default=CHAIN_TYPE, help=f"Chain type (default: {CHAIN_TYPE})"
#     )
#     llm_parser.add_argument(
#         "--chroma_host", type=str, help="Chroma host address", default=CHROMA_DB_HOST
#     )
#     llm_parser.add_argument(
#         "--chroma_port", type=int, help="Chroma port number", default=CHROMA_DB_PORT
#     )
#     llm_parser.add_argument(
#         "--chroma_collection_name",
#         type=str,
#         help="Name of the collection",
#         default=CHROMA_DB_COLLECTION,
#     )
#     llm_parser.add_argument(
#         "--embedding_model_name",
#         default=EMBEDDING_MODEL_NAME,
#         help=f"Embedding model name (default: {EMBEDDING_MODEL_NAME})",
#     )
#     # Command to spin up LLM
#     spin_up_llm_from_mlflow_parser = subparsers.add_parser(
#         "spin_up_llm_from_mlflow", help="Spin up an LLM from MLFlow"
#     )
#     spin_up_llm_from_mlflow_parser.add_argument(
#         "--mlflow_tracking_uri", type=str, help="MLFlow tracking URI"
#     )
#     spin_up_llm_from_mlflow_parser.add_argument(
#         "--mlflow_registry_uri", type=str, help="MLFlow registry URI"
#     )
#     spin_up_llm_from_mlflow_parser.add_argument(
#         "--llm_model_uri", type=str, help="URI of the LLM model in MLFlow"
#     )
#     spin_up_llm_from_mlflow_parser.add_argument(
#         "--embedding_model_uri", type=str, help="URI of the embedding model"
#     )
#     spin_up_llm_from_mlflow_parser.add_argument(
#         "--chroma_host", type=str, help="Chroma host address"
#     )
#     spin_up_llm_from_mlflow_parser.add_argument(
#         "--chroma_port", type=int, help="Chroma port number"
#     )
#     spin_up_llm_from_mlflow_parser.add_argument(
#         "--chroma_collection_name", type=str, help="Name of the collection"
#     )

#     args = parser.parse_args()

#     # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#     callback_manager = CallbackManager([QueueCallback(q)])
#     embedding_model = EmbeddingLoader().load_from_local(model_name=embedding_model_name)

#     qa = LLMLoader().load_from_mlflow(
#         callback_manager=callback_manager,
#         embedding_model_uri=args.embedding_model_uri,
#         chroma_host=args.chroma_host,
#         chroma_port=args.chroma_port,
#         collection_name=args.chroma_collection_name,
#         mlflow_tracking_uri=args.mlflow_tracking_uri,
#         mlflow_registry_uri=args.mlflow_registry_uri,
#         llm_model_uri=args.llm_model_uri,
#     )
#     # TODO abstract away
#     if args.command == "spin_up_llm_from_mlflow":
#         from model_handler import LLMLoader

#         print("ARGUMENTS:")
#         print(args)
#         qa = LLMLoader().load_from_mlflow(
#             callback_manager=callback_manager,
#             embedding_model_uri=args.embedding_model_uri,
#             chroma_host=args.chroma_host,
#             chroma_port=args.chroma_port,
#             collection_name=args.chroma_collection_name,
#             mlflow_tracking_uri=args.mlflow_tracking_uri,
#             mlflow_registry_uri=args.mlflow_registry_uri,
#             llm_model_uri=args.llm_model_uri,
#         )
#         launch_app(qa=qa)

#     if args.command == "spin_up_llm_from_local":
#         print("ARGUMENTS:")
#         print(args)
#         embedding_model = EmbeddingLoader().load_from_local(
#             model_name=args.embedding_model_name
#         )
#         qa = LLMLoader().load_from_local(
#             embedding_model=embedding_model,
#             chroma_host=args.chroma_host,
#             chroma_port=args.chroma_port,
#             collection_name=args.chroma_collection_name,
#             model_type=args.model_type,
#             model_id=args.model_id,
#             model_basename=args.model_basename,
#             context_window_size=args.context_window_size,
#             n_gpu_layers=args.n_gpu_layers,
#             n_batch=args.n_batch,
#             max_new_tokens=args.max_new_tokens,
#             cpu_percentage=args.cpu_percentage,
#             use_memory=args.use_memory,
#             system_prompt=args.system_prompt,
#             chain_type=args.chain_type,
#             callback_manager=callback_manager,
#         )
#         print("QA:", qa)
#         launch_app(qa=qa)
#     print(CHAIN_TYPE)
