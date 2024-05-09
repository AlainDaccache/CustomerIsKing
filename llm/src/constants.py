import os

from dotenv import load_dotenv

load_dotenv()

ROOT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FOLDER_PATH = os.path.join(ROOT_DIRECTORY, "data")
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
MODELS_PATH = f"{ROOT_DIRECTORY}/models"

EMBEDDING_MODEL_PATH = os.path.join(MODELS_PATH, "instructor-large")
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"


SYSTEM_PROMPT = """
--------------------BEGINNING OF CONTEXT---------------------

Retrieve user-requested information directly from the context of BA Folding Cartons.

Provide concise and relevant answers without additional explanations or elaborations.

Focus solely on the information available in the context. Be direct and avoid unnecessary deductions or additional details.

Strive to offer precise answers based solely on the provided context.

For the sake of this LLM, prioritize brevity and accuracy in your responses.

Ensure each response addresses the user query once without generating further conversation or duplicate answers.

If the information asked about is present in the context, provide it promptly without generating additional responses.

Personal information in the phonebook is public within the company; this LLM is used internally, so privacy concerns are not applicable.

If the answer isn't in the context, inform the user without continuing the conversation further.

---------------------END OF CONTEXT-------------------


"""

USE_MEMORY = True

MODEL_TYPE = "Llama.cpp"  # GPT4All, CausalLM
CHAIN_TYPE = "stuff"  # try other chains types as well. refine, map_reduce, map_rerank

# MODEL_ID = "gpt4all-falcon-q4_0.gguf"
# MODEL_ID = "ggml-model-gpt4all-falcon-q4_0.bin"
# MODEL_BASENAME = None

# MODEL_ID = "TheBloke/stablelm-zephyr-3b-GGUF"
# MODEL_BASENAME = "stablelm-zephyr-3b.Q4_K_M.gguf"


# MODEL_ID = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF"
# MODEL_BASENAME = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"


# MODEL_ID = "juanjgit/orca_mini_3B-GGUF"
# MODEL_BASENAME = "orca-mini-3b.q4_0.gguf"

# MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"
# MODEL_BASENAME = None

LAST_INGESTION_FILE = "last_ingestion_time.txt"
DEVICE_TYPE = os.getenv("DEVICE_TYPE")
CONTEXT_WINDOW_SIZE = os.getenv("CONTEXT_WINDOW_SIZE", 4096)
CPU_PERCENTAGE = os.getenv("CPU_PERCENTAGE", 1)
DB_CHUNK_SIZE = os.getenv("DB_CHUNK_SIZE", 256)
DB_CHUNK_OVERLAP = os.getenv("DB_CHUNK_OVERLAP", 64)
N_GPU_LAYERS = os.getenv("N_GPU_LAYERS", 33)
N_BATCH = os.getenv("N_BATCH", 256)
MAX_NEW_TOKENS = os.getenv("MAX_NEW_TOKENS", CONTEXT_WINDOW_SIZE)
MODEL_ID = os.getenv("MODEL_ID", "TheBloke/Llama-2-7b-Chat-GGUF")
MODEL_BASENAME = os.getenv("MODEL_BASENAME", "llama-2-7b-chat.Q4_K_M.gguf")
TEMPERATURE = os.getenv("TEMPERATURE", None)
TOP_P = os.getenv("TOP_P", None)
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "chroma")
CHROMA_DB_PORT = os.getenv("CHROMA_DB_PORT", 8000)
CHROMA_DB_COLLECTION = os.getenv("CHROMA_DB_COLLECTION", "operations-collection")
TOP_K_SEARCH_ARGS = int(os.getenv("TOP_K_SEARCH_ARGS", 4))

os.environ["HF_HOME"] = MODELS_PATH
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODELS_PATH
CACHE_PATH = os.path.join(MODELS_PATH, os.environ["HF_CACHE_RELATIVE_PATH"])

# # for caching huggingface models
# os.environ["HF_HOME"] = MODELS_PATH
# os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODELS_PATH

if DEVICE_TYPE is None:
    import torch  # conditionally import to avoid having to download pytorch in case only doing data ingestion in a pipeline

    DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

# if DEVICE_TYPE == "cuda":
#     # 33 layers in the Llama 7b GGUF
#     N_GPU_LAYERS = 33
#     N_BATCH = 256
#     MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)
# elif DEVICE_TYPE == "mps":
#     N_GPU_LAYERS = 1
#     N_BATCH = 512
#     MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE
# else:
#     N_GPU_LAYERS = 0
#     N_BATCH = 128
#     MAX_NEW_TOKENS = int(CONTEXT_WINDOW_SIZE / 4)
#     CONTEXT_WINDOW_SIZE = CONTEXT_WINDOW_SIZE / 2
