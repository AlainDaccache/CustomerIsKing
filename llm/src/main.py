import gradio as gr
from typing import Any
from queue import Queue, Empty
from langchain.callbacks.base import BaseCallbackHandler
from threading import Thread
from constants import *
from langchain.callbacks.manager import CallbackManager
from model_handler import spin_up_llm


class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()


def launch_app(model_id=MODEL_ID, model_basename=MODEL_BASENAME, context_window_size=CONTEXT_WINDOW_SIZE,
               n_gpu_layers=N_GPU_LAYERS, n_batch=N_BATCH, max_new_tokens=MAX_NEW_TOKENS,
               cpu_percentage=CPU_PERCENTAGE, device_type=DEVICE_TYPE, use_memory=USE_MEMORY,
               system_prompt=SYSTEM_PROMPT, model_type=MODEL_TYPE, chain_type=CHAIN_TYPE,
               embedding_model_name: str = EMBEDDING_MODEL_NAME):
    q = Queue()
    job_done = object()

    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    callback_manager = CallbackManager([QueueCallback(q)])

    qa = spin_up_llm(model_type=model_type,
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
                     callback_manager=callback_manager)

    def answer(question):
        def task():
            res = qa(question)
            answer, docs = res["result"], res["source_documents"]
            s = "\nHere are the relevant sources for this information:\n"

            for i, doc in enumerate(docs):
                s += f"{i + 1}. {doc.metadata['source']}\n"

            q.put(s)
            q.put(job_done)

        t = Thread(target=task)
        t.start()

    s = """
    Welcome to BAI, the intelligent Chatbot for BA Folding Cartons! I am here to assist and guide you through various tasks. Currently, I excel in:
      • Retrieving and synthesizing information from our extensive knowledge base, covering processes, regulations, and more. Feel free to ask questions like, "How do I create a mylar in ArtiosCad?" or "Whose responsibility is it?"
      • Helping with straightforward automation tasks, like creating a standard.

    In the near future, I will also be capable of extracting data from our database, enhancing my capabilities even further.
    """

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot([[None, s]], )
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

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch(share=False,
                debug=False,
                server_name="0.0.0.0",
                ssl_verify=False
                )


if __name__ == "__main__":
    print(CHAIN_TYPE)
    launch_app(model_id=MODEL_ID, model_basename=MODEL_BASENAME, context_window_size=CONTEXT_WINDOW_SIZE,
               n_gpu_layers=N_GPU_LAYERS, n_batch=N_BATCH, max_new_tokens=MAX_NEW_TOKENS,
               cpu_percentage=CPU_PERCENTAGE, device_type=DEVICE_TYPE, use_memory=USE_MEMORY,
               system_prompt=SYSTEM_PROMPT, chain_type=CHAIN_TYPE, model_type=MODEL_TYPE,
               embedding_model_name=EMBEDDING_MODEL_NAME)
