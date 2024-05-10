import gradio as gr
from typing import Any
from queue import Queue, Empty
from langchain.callbacks.base import BaseCallbackHandler
from threading import Thread
from constants import *
from langchain.callbacks.manager import CallbackManager
from model_handler import LLMLoader
from data_handler import EmbeddingLoader


class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()


def launch_app(args):
    q = Queue()
    job_done = object()

    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    callback_manager = CallbackManager([QueueCallback(q)])
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
    print("QA:", qa)
    # TODO abstract away

    def answer(question):
        def task():
            res = qa(question)
            answer, docs = res["result"], res["source_documents"]
            s = "\nHere are the relevant sources for this information:\n"
            unique_sources = list(set([doc.metadata["source"] for doc in docs]))

            for i, doc in enumerate(unique_sources):
                s += f"{i + 1}. {doc}\n"

            q.put(s)
            q.put(job_done)

        t = Thread(target=task)
        t.start()

    s = """
    Welcome to Dunya, the intelligent Chatbot for our Fictional Comany! 
    I am here to assist and guide you through various tasks. Currently, I excel in:
      • Retrieving and synthesizing information from our extensive knowledge base, covering processes, regulations, and more. Feel free to ask questions such as your rights and obligations for an employee or contractor.
      
    In the near future, I will also be capable of extracting data from our database, enhancing my capabilities even further.
    """

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            [[None, s]],
        )
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            question = history[-1][0]
            print("Question: ", question)
            history[-1][1] = ""
            answer(question=question)
            while True:
                try:
                    next_token = q.get(True, timeout=1)
                    if next_token is job_done:
                        break
                    history[-1][1] += next_token
                    yield history
                except Empty:
                    continue

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch(
        share=False,
        debug=False,
        server_name="0.0.0.0",
        server_port=7860,
        ssl_verify=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Command-line tool for spinning up LLM from MLFlow"
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        help="MLFlow tracking URI",
    )
    parser.add_argument(
        "--mlflow_registry_uri",
        type=str,
        help="MLFlow registry URI",
    )
    parser.add_argument(
        "--llm_model_uri",
        type=str,
        help="URI of the LLM model in MLFlow",
    )
    parser.add_argument(
        "--embedding_model_uri",
        type=str,
        help="URI of the embedding model",
    )
    parser.add_argument(
        "--chroma_host",
        type=str,
        help="Chroma host address",
    )
    parser.add_argument(
        "--chroma_port",
        type=int,
        help="Chroma port number",
    )
    parser.add_argument(
        "--chroma_collection_name",
        type=str,
        help="Name of the collection",
    )

    args = parser.parse_args()

    launch_app(args)
