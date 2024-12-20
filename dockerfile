FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 as base

# Update PATH to include conda
ENV PATH="/miniconda/bin:$PATH"

# Set CUDA architecture flags
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Set the working directory
WORKDIR /app

# Install vim, git, bash
RUN apt-get update && apt-get install -y vim \
    git \
    bash \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

FROM base as teseract
# OCR Tesseract dependencies
RUN apt-get update && apt-get install -y tesseract-ocr

## Developer Tools
#RUN apt-get update && apt-get install -y libtesseract-dev

RUN apt-get update && apt-get install -y g++ \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libpng-dev \
    libjpeg8-dev \
    libtiff5-dev \
    zlib1g-dev \
    libleptonica-dev \
    libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

### help using command line
# RUN apt-get update && apt-get install -y snap && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*
# RUN snap install --channel=edge tesseract

### Install language pack
RUN apt-get update && apt-get install -y tesseract-ocr-eng \
    tesseract-ocr-chi-tra && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


FROM teseract as conda

# Install miniconda
# Install dependencies and miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh && \
    bash Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -b -p /miniconda && \
    rm Miniconda3-py310_24.7.1-0-Linux-x86_64.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda init

FROM conda as baseline

RUN conda init
# Create conda environment and install GPU versions with CUDA 11.8
# Create conda environment and install GPU versions with CUDA 11.8
RUN conda config --set ssl_verify false
RUN conda update --all -y && \
    conda install -c anaconda ca-certificates -y && \
    conda install -c anaconda openssl -y

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
    conda config --set ssl_verify false

RUN conda create -n baseline python=3.10 -y

# pyserini dependencies
RUN conda run -n baseline conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=12.1 -c pytorch -c nvidia -y
RUN conda run -n baseline conda install -c conda-forge openjdk=21 maven lightgbm nmslib -y 
RUN conda run -n baseline conda install -c pytorch -c nvidia faiss-gpu=1.8.0 pytorch=*=*cuda* pytorch-cuda=12 -y
RUN conda run -n baseline conda install -c conda-forge numpy==1.23.5 scipy==1.9.3 scikit-learn==1.5.2 -y
RUN conda run -n baseline pip install pdfplumber tqdm rank_bm25 jieba pillow pymupdf loguru scikit-learn
RUN conda run -n baseline pip install langchain langchain-core langchain-community langchain-huggingface langchain-text-splitters
RUN conda run -n baseline pip install pyserini ollama marqo gensim
RUN conda run -n baseline pip install  python-dotenv
# Install pytesseract
RUN conda run -n baseline pip install pytesseract

# Activate conda environment by default
RUN echo "conda activate baseline" >> ~/.bashrc
