import torch
import os
from langchain.llms import HuggingFacePipeline
import multiprocessing
from constants import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODELS_PATH,
    DEVICE_TYPE,
    N_BATCH,
    CONTEXT_WINDOW_SIZE,
    N_GPU_LAYERS,
    CPU_PERCENTAGE,
    MODEL_TYPE
)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from langchain.llms import GPT4All
from langchain.callbacks.manager import CallbackManager

import logging

from constants import MODEL_ID, EMBEDDING_MODEL_NAME, MODEL_BASENAME, SYSTEM_PROMPT, USE_MEMORY, TEMPERATURE, TOP_P, \
    CHAIN_TYPE
from data_handler import ingest, load_db

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig,
)


def load_llamacpp_model(callback_manager: CallbackManager = None,
                        model_id=MODEL_ID,
                        model_basename=MODEL_BASENAME,
                        device_type=DEVICE_TYPE,
                        context_window_size=CONTEXT_WINDOW_SIZE,
                        max_new_tokens=MAX_NEW_TOKENS,
                        n_gpu_layers=N_GPU_LAYERS,
                        cpu_percentage=CPU_PERCENTAGE,
                        n_batch=N_BATCH,
                        temperature=TEMPERATURE,
                        top_p=TOP_P
                        ):
    logging.info("Using Llamacpp for GGUF quantized models")
    model_path = hf_hub_download(
        repo_id=model_id,
        filename=model_basename,
        resume_download=True,
        cache_dir=MODELS_PATH,
    )

    kwargs = {
        "model_path": model_path,
        "temperature": 0.75,
        "top_p": 1,
        "n_ctx": context_window_size,
        "max_tokens": max_new_tokens,
        "n_batch": n_batch,  # set this based on your GPU & CPU RAM
        "callback_manager": callback_manager,
        "verbose": True  # required for callback manager streaming?
    }
    if temperature:
        kwargs["temperature"] = temperature
    if top_p:
        kwargs["top_p"] = top_p

    if device_type.lower() == "cpu":
        kwargs["n_threads"] = cpu_percentage * multiprocessing.cpu_count()
    if device_type.lower() == "mps":
        kwargs["n_gpu_layers"] = 1
    if device_type.lower() == "cuda":
        kwargs["n_gpu_layers"] = n_gpu_layers  # set this based on your GPU
        kwargs["n_threads"] = 1 # full offloading?
    return LlamaCpp(**kwargs)


def load_hf_causallm(model_id: str = MODEL_ID,
                     device_type: str = DEVICE_TYPE,
                     temperature: float = TEMPERATURE,
                     top_p: float = TOP_P,
                     max_new_tokens: int = MAX_NEW_TOKENS,
                     callback_manager: CallbackManager = None):
    if device_type.lower() in ["mps", "cpu"]:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id, cache_dir=MODELS_PATH)
        model = LlamaForCausalLM.from_pretrained(model_id, cache_dir=MODELS_PATH)
    else:
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODELS_PATH)
        logging.info("Tokenizer loaded")
        # TODO use bitsandbytes, dynamically find max memory etc.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=MODELS_PATH,
            trust_remote_code=True,  # set these if you are using NVIDIA GPU
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            max_memory={
                0: "15GB"
            },  # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()

    generation_config = GenerationConfig.from_pretrained(model_id)
    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


def load_gpt4all_model(model_id,
                       callback_manager: CallbackManager = None
                       ):
    n_threads = CPU_PERCENTAGE * multiprocessing.cpu_count()

    llm = GPT4All(
        model=os.path.join(MODELS_PATH, model_id),
        callbacks=callback_manager,
        # n_ctx=CONTEXT_WINDOW_SIZE,
        backend='gptj',
        # n_batch=N_BATCH,
        verbose=True,
        n_threads=n_threads,
    )
    return llm


def load_model(model_type: str = MODEL_TYPE,
               model_id: str = MODEL_ID,
               model_basename: str = MODEL_BASENAME,
               device_type: str = DEVICE_TYPE,
               context_window_size: str = CONTEXT_WINDOW_SIZE,
               cpu_percentage: float = CPU_PERCENTAGE,
               n_gpu_layers: int = N_GPU_LAYERS,
               n_batch: int = N_BATCH,
               max_new_tokens: int = MAX_NEW_TOKENS,
               temperature: float = TEMPERATURE,
               top_p: float = TOP_P,
               callback_manager: CallbackManager = None):
    if model_type == "Llama.cpp":
        llm = load_llamacpp_model(
            model_id=model_id,
            model_basename=model_basename,
            device_type=device_type,
            context_window_size=context_window_size,
            cpu_percentage=cpu_percentage,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            callback_manager=callback_manager
        )
    elif model_type == "GPT4All":
        llm = load_gpt4all_model(model_id=model_id, callback_manager=callback_manager)
        return llm
    elif model_type == "CausalLM":
        llm = load_hf_causallm(model_id=model_id,
                               device_type=device_type,
                               max_new_tokens=max_new_tokens,
                               top_p=top_p,
                               temperature=temperature,
                               callback_manager=callback_manager)
    else:
        raise Exception("Not supported yet")
    return llm


def generate_prompt_template(
        model_id: str = MODEL_ID,
        system_prompt: str = SYSTEM_PROMPT,
        use_memory: str = USE_MEMORY):
    if use_memory:
        instruction = """
            Context: {history} \n {context}
            User: {question}"""
    else:
        instruction = """
            Context: {context} \n
            User: {question}"""

    prompt_template_map = {
        "TheBloke/stablelm-zephyr-3b-GGUF": f"<|user|>\n {system_prompt} '\n\n' {instruction} <|endoftext|> \n <|assistant|>",
        "TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF": f"""
                                                    <|im_start|>system
                                                    {system_prompt}<|im_end|>
                                                    <|im_start|>user
                                                    {instruction}<|im_end|>
                                                    <|im_start|>assistant
                                                    """,
        "TheBloke/Llama-2-7b-Chat-GGUF": f"""
            [INST] 
                <<SYS>>
                {system_prompt}
                <</SYS>>
                {instruction}
            [/INST]
        """
    }
    if model_id in prompt_template_map:
        prompt_template = prompt_template_map[model_id]
    else:
        print(f"Model ID ({model_id}) not in prompt template map. Using default template")
        prompt_template = f"<System> {system_prompt} </System> \n <User> {instruction} </User>"

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"], template=prompt_template
    )
    return prompt


def generate_retrieval_qa(llm,
                          prompt,
                          use_memory,
                          retriever,
                          callback_manager,
                          chain_type):
    chain_type_kwargs = {"prompt": prompt}
    if use_memory:
        chain_type_kwargs["memory"] = ConversationBufferMemory(input_key="question", memory_key="history")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        # verbose=True,
        callbacks=callback_manager,
        chain_type_kwargs=chain_type_kwargs,
    )
    return qa


def spin_up_llm(callback_manager: CallbackManager = None, model_id=MODEL_ID, model_basename=MODEL_BASENAME,
                context_window_size=CONTEXT_WINDOW_SIZE,
                n_gpu_layers=N_GPU_LAYERS, n_batch=N_BATCH, max_new_tokens=MAX_NEW_TOKENS,
                cpu_percentage=CPU_PERCENTAGE, device_type=DEVICE_TYPE, use_memory=USE_MEMORY,
                system_prompt=SYSTEM_PROMPT, model_type=MODEL_TYPE, chain_type=CHAIN_TYPE,
                embedding_model_name: str = EMBEDDING_MODEL_NAME):
    print("Device type:", device_type)
    retriever = load_db(embedding_model_name=embedding_model_name)
    local_llm = load_model(model_type=model_type,
                           model_id=model_id,
                           model_basename=model_basename,
                           n_batch=n_batch,
                           n_gpu_layers=n_gpu_layers,
                           max_new_tokens=max_new_tokens,
                           context_window_size=context_window_size,
                           cpu_percentage=cpu_percentage,
                           device_type=device_type,
                           callback_manager=callback_manager
                           )
    prompt = generate_prompt_template(model_id=model_id,
                                      system_prompt=system_prompt,
                                      use_memory=use_memory)

    qa = generate_retrieval_qa(llm=local_llm,
                               prompt=prompt,
                               retriever=retriever,
                               callback_manager=callback_manager,
                               use_memory=use_memory,
                               chain_type=chain_type)
    return qa
