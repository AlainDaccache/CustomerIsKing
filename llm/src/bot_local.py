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


def launch_app(
    chroma_host,
    chroma_port,
    chroma_collection_name,
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
    q = Queue()
    job_done = object()

    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    callback_manager = CallbackManager([QueueCallback(q)])
    embedding_model = EmbeddingLoader().load_from_local(model_name=embedding_model_name)
    qa = LLMLoader().load_from_local(
        embedding_model=embedding_model,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        collection_name=chroma_collection_name,
        model_type=model_type,
        model_id=model_id,
        model_basename=model_basename,
        context_window_size=context_window_size,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        max_new_tokens=max_new_tokens,
        cpu_percentage=cpu_percentage,
        use_memory=use_memory,
        system_prompt=system_prompt,
        chain_type=chain_type,
        callback_manager=callback_manager,
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
    from constants import *

    print(CHAIN_TYPE)
    launch_app(
        chroma_host=CHROMA_DB_HOST,
        chroma_port=CHROMA_DB_PORT,
        chroma_collection_name=CHROMA_DB_COLLECTION,
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
        chain_type=CHAIN_TYPE,
        model_type=MODEL_TYPE,
        embedding_model_name=EMBEDDING_MODEL_NAME,
    )
