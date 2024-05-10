import mlflow
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from constants import MODELS_PATH, ROOT_DIRECTORY
from huggingface_hub import hf_hub_download, snapshot_download
from mlflow.pyfunc import log_model, load_model

import logging
import os


def download_and_register_llm(
    hugging_face_repo_id,
    mlflow_registry_uri,
    mlflow_tracking_uri,
    hugging_face_filename=None,
    **kwargs,
):
    print("Hugging Face Repo ID:", hugging_face_repo_id)
    print("Hugging Face Filename:", hugging_face_filename)
    print("Keyword Args:", kwargs)
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.path.join(ROOT_DIRECTORY, ".env"))
    print(f"Downloading model {hugging_face_repo_id} from Hugging Face Hub!")

    if hugging_face_filename:
        model_path = hf_hub_download(
            repo_id=hugging_face_repo_id,
            filename=hugging_face_filename,
            resume_download=True,
            # cache_dir=local_models_dir,
        )
    else:
        model_path = hf_hub_download(
            repo_id=hugging_face_repo_id,
            resume_download=True,
            # cache_dir=local_models_dir,
        )
    print(f"Downloaded model {hugging_face_repo_id} from Hugging Face Hub!")
    print("Model path is:", model_path)
    # Because callback manager is fed into
    if not kwargs:
        kwargs = {
            "model_path": model_path,
            "temperature": 0.75,
            "top_p": 1,
            "n_ctx": 4096,
            "max_tokens": 4096,
            "n_batch": 256,
            # "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
            "verbose": True,
            "n_gpu_layers": 33,
            "n_threads": 1,
        }

    mlflow_endpoint_url = os.getenv("MLFLOW_TRACKING_URI")
    print("MLFlow Endpoint URL:", mlflow_endpoint_url)
    mlflow.set_tracking_uri(mlflow_endpoint_url)
    experiment_name = "llm_bot"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        from langchain.llms import LlamaCpp
        from langchain.chains import RetrievalQA
        from langchain.memory import ConversationBufferMemory
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain.prompts import PromptTemplate

        class LLMWrapper(mlflow.pyfunc.PythonModel):
            """Model object will be acccessible via model attr"""

            def gen_prompt_template(self, system_prompt, use_memory):
                if use_memory:
                    instruction = """
                        Context: {history} \n {context}
                        User: {question}"""
                else:
                    instruction = """
                        Context: {context} \n
                        User: {question}"""
                prompt_template = f"""
                    [INST] 
                        <<SYS>>
                        {system_prompt}
                        <</SYS>>
                        {instruction}
                    [/INST]
                """
                prompt = PromptTemplate(
                    input_variables=["history", "context", "question"],
                    template=prompt_template,
                )
                return prompt

            def generate_retrieval_qa(
                self,
                llm,
                system_prompt,
                use_memory,
                retriever,
                callback_manager,
                chain_type,
            ):

                prompt = self.gen_prompt_template(
                    use_memory=use_memory, system_prompt=system_prompt
                )
                print("Prompt:", prompt)
                chain_type_kwargs = {"prompt": prompt}
                if use_memory:
                    chain_type_kwargs["memory"] = ConversationBufferMemory(
                        input_key="question", memory_key="history"
                    )
                print("DEBUG:")
                print(llm)
                print(chain_type)
                print(retriever)
                print(callback_manager)
                print(chain_type_kwargs)
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

            def load_context(self, context):
                """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
                Args:
                    context: MLflow context where the model artifact is stored.
                """

                # model_path = context.artifacts["model_path"]

                # optimized parameters
                llama_config = context.model_config["llama_config"]
                llama_config["model_path"] = context.artifacts["model_path"]
                print("Llama Config Model Path:", llama_config["model_path"])
                qa_config = context.model_config["qa_config"]

                # config for bot
                system_prompt = qa_config["system_prompt"]
                self.top_k = qa_config.get("top_k", 4)
                chain_type = qa_config.get("chain_type", "stuff")

                self.llama_model_lambda = lambda callback_manager: LlamaCpp(
                    callback_manager=callback_manager, **llama_config
                )

                self.qa_retriever_lambda = lambda callback_manager, retriever, use_memory: self.generate_retrieval_qa(
                    llm=self.llama_model_lambda(callback_manager=callback_manager),
                    system_prompt=system_prompt,
                    retriever=retriever,
                    callback_manager=callback_manager,
                    use_memory=use_memory,
                    chain_type=chain_type,
                )

            def get_retriever(
                self,
                embedding_model_uri,
                chroma_host,
                chroma_port,
                collection_name,
                device_type,
            ):
                from langchain.vectorstores import Chroma
                from mlflow.pyfunc import load_model
                from chromadb import HttpClient
                from chromadb.config import Settings

                embedding_mlflow = load_model(
                    model_uri=embedding_model_uri
                ).unwrap_python_model()

                embedding_function = embedding_mlflow.model_lambda(
                    device_type=device_type
                )
                client = HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=Settings(
                        allow_reset=True,
                        anonymized_telemetry=False,
                        is_persistent=True,
                    ),
                )
                chroma = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=embedding_function,
                )
                retriever = chroma.as_retriever(search_kwargs={"k": self.top_k})
                return retriever

            def init_qa_bot(
                self,
                callback_manager,
                embedding_model_uri,
                chroma_host,
                chroma_port,
                collection_name,
                device_type,
                use_memory=True,
            ):
                retriever = self.get_retriever(
                    chroma_host=chroma_host,
                    chroma_port=chroma_port,
                    embedding_model_uri=embedding_model_uri,
                    collection_name=collection_name,
                    device_type=device_type,
                )
                self.callback_manager = callback_manager
                self.retriever = retriever
                self.qa_retriever = self.qa_retriever_lambda(
                    callback_manager=callback_manager,
                    retriever=retriever,
                    use_memory=use_memory,
                )

                return self.qa_retriever

            def generate_tokens_queue(self, input):
                return "Great Success!"

            def generate_tokens_stdout(self, input):
                return self.qa_retriever(input)

            def predict(self, context, input):
                if isinstance(self.callback_manager, StreamingStdOutCallbackHandler):
                    return self.generate_tokens_stdout(input)
                else:
                    raise Exception(f"Callback manager not supported yet")

        input_schema = Schema(
            [
                ColSpec(type="string", name="input_text"),
            ]
        )
        output_schema = Schema([ColSpec(type="string", name="result")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        prompt_filename = kwargs.get("prompt_filename", "prompt-v1.txt")
        prompt_path = os.path.join(ROOT_DIRECTORY, "prompts", prompt_filename)
        print("Prompt Path:", prompt_path)
        artifacts = {"model_path": kwargs["model_path"]}

        with open(file=prompt_path, mode="r") as fp:
            system_prompt = fp.read()  # " ".join(line.rstrip() for line in fp)
        print("System Prompt:", system_prompt)

        model_config = {
            "llama_config": kwargs,
            "qa_config": {
                "system_prompt": system_prompt,
                "top_k": 4,
                "chain_type": "stuff",
            },
        }

        conda_spec_path = os.path.join(ROOT_DIRECTORY, "conda.yml")
        from mlflow.pyfunc import log_model

        model_info = log_model(
            artifact_path="model",
            python_model=LLMWrapper(),
            artifacts=artifacts,
            signature=signature,
            input_example={
                "input_text": "How many sales have we made in the past year in the UK?"
            },
            conda_env=conda_spec_path,
            # code_paths=[os.path.join(ROOT_DIRECTORY, "requirements.txt")],
            model_config=model_config,
        )

        # from langchain.llms import LlamaCpp
        # from langchain.chains import RetrievalQA
        # from langchain.memory import ConversationBufferMemory
        # from langchain.callbacks.manager import CallbackManager
        # from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        # import pysqlite3
        # import sys

        # sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
        # # llm_run_uri = "runs:/26ea8cf27f684bd395198ad4d0e2eb29/model"
        # llm_run_uri = model_info.model_uri
        # print("Model URI:", llm_run_uri)

        # loaded_llm_mlflow = load_model(
        #     model_uri=llm_run_uri
        # ).unwrap_python_model()  # to retrieve our class

        # chroma_host = "chroma"
        # chroma_port = "8000"
        # collection_name = "operations-collection"
        # embedding_model_run_id = "ebdbabdb63784baf9160a4d86ab2a371"

        # embedding_model_uri = "runs:/{}/model".format(embedding_model_run_id)

        # qa_retriever = loaded_llm_mlflow.init_qa_bot(
        #     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        #     embedding_model_uri=embedding_model_uri,
        #     chroma_host=chroma_host,
        #     chroma_port=chroma_port,
        #     collection_name=collection_name,
        #     device_type=
        # )
        # # print(qa_retriever)


def download_and_register_embeddings(
    hugging_face_repo_id,
    mlflow_registry_uri,
    mlflow_tracking_uri,
    hugging_face_filename=None,
):
    """
    repo_id and filename in HuggingFaceHub
    """

    os.environ["HF_HOME"] = MODELS_PATH
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODELS_PATH
    print(f"Downloading model {hugging_face_repo_id} from Hugging Face Hub!")

    if hugging_face_filename:
        downloaded_path = hf_hub_download(
            repo_id=hugging_face_repo_id,
            filename=hugging_face_filename,
            resume_download=True,
            # cache_dir=local_models_dir,
        )
    else:
        downloaded_path = snapshot_download(
            repo_id=hugging_face_repo_id,
            resume_download=True,
            # cache_dir=local_models_dir,
        )

    print(f"Downloaded model from Hugging Face Hub into path: {downloaded_path}")
    artifacts = {"model_path": downloaded_path}
    mlflow_endpoint_url = os.getenv("MLFLOW_TRACKING_URI")
    print("MLFlow Endpoint URL:", mlflow_endpoint_url)
    mlflow.set_tracking_uri(mlflow_endpoint_url)
    from dotenv import load_dotenv

    load_dotenv()
    experiment_name = "embedding_models"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):

        class EmbeddingWrapper(mlflow.pyfunc.PythonModel):
            """Model object will be acccessible via model attr"""

            def load_context(self, context):
                """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
                Args:
                    context: MLflow context where the model artifact is stored.
                """
                print("Context:", context)
                print("Context Model Config:", context.model_config)
                model_path = context.artifacts["model_path"]
                print("Model Path:", model_path)
                # device_type = "cuda" if torch.cuda.is_available() else "cpu"
                # print("Device Type:", device_type)
                self.model_lambda = lambda device_type: HuggingFaceInstructEmbeddings(
                    model_name=model_path,
                    model_kwargs={"device": device_type},
                )

        conda_spec_path = os.path.join(ROOT_DIRECTORY, "conda.yml")
        logged_model_path = "model"  # hugging_face_repo_id.replace("/", "--")
        from mlflow.pyfunc import log_model, load_model

        model_info = log_model(
            artifact_path=logged_model_path,  # logged_model_path,
            python_model=EmbeddingWrapper(),
            artifacts=artifacts,
            # code_paths=[os.path.join(ROOT_DIRECTORY, "requirements.txt")],
            conda_env=conda_spec_path,
        )
        # run_id = mlflow.active_run().info.run_id
        # mlflow.register_model(f"runs:/{run_id}/{logged_model_path}")

        loaded_model = load_model(model_uri=model_info.model_uri)
        print(loaded_model)


# download_and_register_llm(
#     hugging_face_repo_id="TheBloke/Llama-2-7b-Chat-GGUF",
#     hugging_face_filename="llama-2-7b-chat.Q4_K_M.gguf",
# )

# download_and_register_llm(hugging_face_repo_id=None, hugging_face_filename=None)

# download_and_register_embeddings(
#     hugging_face_repo_id="hkunlp/instructor-large",
# )

# download_and_register_llm(
#     hugging_face_repo_id="TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF",
#     hugging_face_filename="tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
# )


# import mlflow
# import os
# from dotenv import load_dotenv

# load_dotenv()
# mlflow_endpoint_url = os.getenv("MLFLOW_TRACKING_URI")
# print("MLFlow Endpoint URL:", mlflow_endpoint_url)
# mlflow.set_tracking_uri(mlflow_endpoint_url)
# model_name = "embeddings-model"
# model_version = 3
# model_uri = f"models:/{model_name}/{model_version}"
# model_config = {"device_type": "cuda"}
# model = mlflow.pyfunc.load_model(model_uri, model_config=model_config)
