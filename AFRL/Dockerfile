# Use a Pytorch base image with CUDA and cudnn support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# https://stackoverflow.com/a/63377623
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    ninja-build \
    && apt-get install -y ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*


# Set CUDA_HOME
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV FORCE_CUDA=1

# Clone the HIPIE repository
RUN git clone https://github.com/shankar-veludandi/HIPIE /app/HIPIE

# Change to the HIPIE directory
WORKDIR /app/HIPIE

# Copy requirements.txt file into the Docker image
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install detectron, panopticapi, and SAM
RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git
RUN pip3 install --no-cache-dir git+https://github.com/cocodataset/panopticapi.git
RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Clone and set up the YouTube-VOS cocoapi repository
RUN git clone https://github.com/youtubevos/cocoapi.git /app/HIPIE/cocoapi
WORKDIR /app/HIPIE/cocoapi/PythonAPI
RUN wget https://github.com/youtubevos/cocoapi/raw/master/PythonAPI/pycocotools/ytvos.py && \
    wget https://github.com/youtubevos/cocoapi/raw/master/PythonAPI/pycocotools/ytvoseval.py && \
    mkdir -p /usr/local/lib/python3.10/dist-packages/pycocotools/ && \
    cp ytvos.py /usr/local/lib/python3.10/dist-packages/pycocotools/ && \
    cp ytvoseval.py /usr/local/lib/python3.10/dist-packages/pycocotools/

# Compile MultiScaleDeformableAttention
WORKDIR /app/HIPIE/projects/HIPIE/hipie/models/maskdino/pixel_decoder/ops
RUN python3 setup.py build_ext --inplace

# Change back to the HIPIE directory
WORKDIR /app/HIPIE

# Add the module to the Python path
ENV PYTHONPATH="/app/HIPIE/projects/HIPIE/hipie/models/maskdino/pixel_decoder/ops:${PYTHONPATH}"

# Create the weights directory and download the required weights
RUN mkdir -p /app/HIPIE/weights
WORKDIR /app/HIPIE/weights
RUN wget -c https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl

# Download and setup SAM weights
RUN wget -c https://huggingface.co/n0r9st/segment-anything/resolve/main/sam_vit_h_4b8939.pth

# Create the directory for the BERT-base model and download necessary files
RUN mkdir -p /app/HIPIE/projects/HIPIE/bert-base-uncased
WORKDIR /app/HIPIE/projects/HIPIE/bert-base-uncased
RUN wget -c https://huggingface.co/bert-base-uncased/resolve/main/config.json && \
    wget -c https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt && \
    wget -c https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin

# Change back to the HIPIE directory
WORKDIR /app/HIPIE

# Default command
CMD ["python3", "demo.py"]
