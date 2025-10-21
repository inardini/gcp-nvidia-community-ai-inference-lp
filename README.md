# AI Inference Learning Path

A hands-on learning path for understanding AI model inference on GPUs, developed in collaboration with Google Cloud and NVIDIA.

## Overview

This repository provides interactive Jupyter notebooks that teach the fundamentals of running AI models in production. You'll learn how to optimize inference performance, measure critical metrics, and make informed trade-offs between latency and throughput.

## Learning Path Structure

### Learning Path 01: Foundations of Inference

**[01 - Foundations of Inference](learning_path_01/01_foundations_of_inference.ipynb)**

- CPU vs GPU inference comparison
- Understanding the inference pipeline (preprocessing, batching, forward pass, decoding)
- Token-by-token generation mechanics
- Extending inference concepts to other modalities (images, audio, video)

**[02 - Performance Metrics](learning_path_01/02_performance_metrics.ipynb)**

- Key metrics: latency, throughput, TTFT (Time to First Token), ITL (Inter-Token Latency)
- Batch size optimization and the throughput-latency trade-off
- Finding the "sweet spot" for your use case
- Impact of sequence length and sampling parameters
- Practical UX considerations for production deployments

## Prerequisites

- Hugging Face account with access to gated models:
  - [Gemma 3 270M Instruction-Tuned](https://huggingface.co/google/gemma-3-270m-it)
  - [NVIDIA Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- GPU access (provided free via Google Colab, or use your own NVIDIA GPU)
- Basic familiarity with Python and Jupyter notebooks

## Quick Start

### Option 1: Google Colab (Recommended)

The easiest way to get started is with Google Colab, which provides free GPU access:

1. Open [01_foundations_of_inference.ipynb](https://colab.research.google.com/github/inardini/gcp-nvidia-community-ai-inference-lp/blob/main/learning_path_01/01_foundations_of_inference.ipynb) in Colab
2. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU
3. Enter your Hugging Face token when prompted
4. Run the cells in order

Then proceed to [02_performance_metrics.ipynb](https://colab.research.google.com/github/inardini/gcp-nvidia-community-ai-inference-lp/blob/main/learning_path_01/02_performance_metrics.ipynb)

### Option 2: Local Jupyter with Docker

If you have an NVIDIA GPU available locally:

```bash
cd learning_path_01

docker run -it --rm \
  --gpus all \
  --shm-size=16g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.05-py3 \
  bash -lc 'set -e
    python -m pip install -U pip
    pip install -U "vllm>=0.10.1" "transformers>=4.56" matplotlib ipywidgets huggingface_hub accelerate
    jupyter lab --ip=0.0.0.0 --no-browser --allow-root'
```

Access Jupyter Lab at `http://localhost:8888` and open the notebooks in order.

## What You'll Learn

- **Performance fundamentals**: Why GPUs are essential for inference and how to measure their efficiency
- **Practical optimization**: How to tune batch size, sequence length, and sampling parameters
- **Production trade-offs**: Balancing system capacity (throughput) with user experience (latency)
- **Benchmarking methodology**: Reproducible experiments using p50/p90 metrics
- **Real-world considerations**: Choosing the right configuration for chat, summarization, RAG, and other applications

## Key Concepts

- **Latency**: Time from prompt submission to complete response
- **Throughput**: Tokens generated per second (system capacity)
- **TTFT**: Time to first token (perceived responsiveness)
- **ITL**: Inter-token latency (streaming speed)
- **Batch size**: Number of concurrent requests processed together
- **KV cache**: Memory optimization for autoregressive generation

## Running on Google Cloud

For optimal performance, consider running on Google Cloud with NVIDIA GPUs:

- Compute Engine VMs with A100, L4, or H100 GPUs
- Vertex AI Workbench for managed Jupyter environments
- GKE with GPU node pools for production serving

## Contributing

This is an educational resource. Issues and suggestions are welcome to improve clarity and accuracy.

## License

You can find the license [here](./LICENSE)
