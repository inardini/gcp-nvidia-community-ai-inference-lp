# Intro to Inference: How to Run AI Models on a GPU

## Setup

```bash
cd learning_path_01

docker run -it --rm \
  --gpus all \
  --shm-size=16g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\
  -p 8888:8888 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.05-py3 \
  bash -lc 'set -e
    python -m pip install -U pip
    pip install -U "vllm>=0.10.1" "transformers>=4.56" matplotlib ipywidgets huggingface_hub accelerate
    jupyter lab --ip=0.0.0.0 --no-browser --allow-root'
  ```