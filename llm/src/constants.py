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
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Be concise.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.

You will use the provided context to answer user questions.
Read the given context before answering questions and think step by step.
Think critically about whether you need the context to answer those user questions.
For instance, if the user gives you a simple greeting like a 'Hi, how are you?',
you probably don't need the context.
So think critically about whether you need to
use the given context.
One tip, if the question seems technical, then you probably
need to use the context.
Please be clear and concise, to the point. use bullet points if you think
it's deemed useful, for instance when asked about a procedure/method.
 Be concise.
And finally, I remind you, don't repeat the user's question, and be concise and clear, to the point.
DONT' MENTION ANYTHING IN THE ANSWER RELATED TO WHAT I TOLD YOU IN THIS CONTEXT, THIS IS JUST
FOR YOUR INFORMATION AND HOW TO THINK ABOUT YOUR ANSWER

 Be concise. Be concise. Be concise. Be concise. Be concise. Be concise. Be concise. Be concise.
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
DB_CHUNK_SIZE = os.getenv("DB_CHUNK_SIZE", 512)
DB_CHUNK_OVERLAP = os.getenv("DB_CHUNK_OVERLAP", 256)
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
