# FROM python:3.10.6
# FROM nvidia/cuda:11.6.2-base-ubuntu20.04
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

# Install Python
# RUN apt-get update && \
#     apt-get install -y python3-pip python3-dev python-is-python3 && \
#     rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# LLM specific binaries
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y libgl1 && \
    apt-get install -y poppler-utils && \
    apt-get install -y libtesseract-dev && \
    apt-get install -y tesseract-ocr

RUN curl -sSL https://install.python-poetry.org | python3.10 - --preview
RUN pip3 install --upgrade requests
RUN ln -fs /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

# RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
# RUN python -m pip install pysqlite3-binary openpyxl langchain-community gradio accelerate bitsandbytes kaleido python-multipart langchain chromadb langchainhub bs4 InstructorEmbedding sentence-transformers==2.2.2 docx2txt gpt4all unstructured[pdf]
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose

COPY . .

# CMD ["python", "-m", "main"]
CMD ["/bin/bash"]