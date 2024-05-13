from fastapi import FastAPI
from model_handler import LLMLoader
from data_handler import EmbeddingLoader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamingStdOutCallbackHandler
from constants import *

app = FastAPI()

# Set up callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Load embedding model
embedding_model = EmbeddingLoader().load_from_local(model_name=EMBEDDING_MODEL_NAME)
# Load LLMLoader
qa = LLMLoader().load_from_local(
    embedding_model=embedding_model,
    chroma_host=CHROMA_DB_HOST,
    chroma_port=CHROMA_DB_PORT,
    collection_name=CHROMA_DB_COLLECTION,
    model_id=MODEL_ID,
    model_basename=MODEL_BASENAME,
    context_window_size=CONTEXT_WINDOW_SIZE,
    n_gpu_layers=N_GPU_LAYERS,
    n_batch=N_BATCH,
    max_new_tokens=MAX_NEW_TOKENS,
    cpu_percentage=CPU_PERCENTAGE,
    use_memory=USE_MEMORY,
    system_prompt=SYSTEM_PROMPT,
    chain_type=CHAIN_TYPE,
    model_type=MODEL_TYPE,
    callback_manager=callback_manager,
)


@app.get("/answer/")
async def answer(question: str):
    print("Received Question!\n", question)
    res = qa(question)
    answer, docs = res["result"], res["source_documents"]
    if len(docs):
        s = "\n\nHere are the relevant sources for this information:\n"
        unique_sources = list(set([doc.metadata["source"] for doc in docs]))
        for i, doc in enumerate(unique_sources):
            s += f"• {doc}\n"

        answer = f"{answer}{s}"

    return {"response": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6060)
