# Learning Path 01: Foundations of AI Inference

This learning path introduces the fundamentals of running AI models on GPUs, covering both the mechanics of inference and how to optimize performance for production workloads.

## Notebooks

### 01 - Foundations of Inference
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inardini/gcp-nvidia-community-ai-inference-lp/blob/main/learning_path_01/01_foundations_of_inference.ipynb)

Learn the core concepts of model inference:
- **CPU vs GPU**: Understand why hardware acceleration matters through hands-on benchmarking
- **Inference Pipeline**: Step through preprocessing, batching, forward pass, and decoding
- **Token Generation**: See how autoregressive models generate text one token at a time
- **Cross-Modal Concepts**: Apply inference understanding to images, audio, and video

**Model**: Gemma 3 270M Instruction-Tuned

### 02 - Performance Metrics
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inardini/gcp-nvidia-community-ai-inference-lp/blob/main/learning_path_01/02_performance_metrics.ipynb)

Master inference performance optimization:
- **Key Metrics**: Latency, throughput, TTFT (Time to First Token), ITL (Inter-Token Latency)
- **Batch Size Optimization**: Find the sweet spot between capacity and responsiveness
- **Comprehensive Benchmarking**: Understand p50 vs p90 percentiles and tail latency
- **Parameter Impact**: Measure how sequence length and sampling affect performance
- **Production Trade-offs**: Choose the right configuration for your use case

**Model**: NVIDIA Nemotron Nano 9B v2

## Running the Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge on any notebook above. Colab provides free GPU access (T4) with the option to upgrade to more powerful GPUs.

**Advantages:**
- No local setup required
- Free GPU access
- Pre-installed dependencies
- Easy sharing and collaboration

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

Access Jupyter Lab at `http://localhost:8888`

## Prerequisites

- **Hugging Face Account**: You'll need access tokens for gated models
  - [Gemma 3 270M IT](https://huggingface.co/google/gemma-3-270m-it)
  - [NVIDIA Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- **GPU**: Colab provides this, or use local NVIDIA GPU with CUDA support
- **Basic Python**: Familiarity with PyTorch and Jupyter notebooks

## What You'll Build

By the end of this learning path, you'll be able to:
- ✅ Benchmark inference performance accurately using industry-standard metrics
- ✅ Optimize batch size and parameters for your specific latency requirements
- ✅ Understand the throughput-latency trade-off in production systems
- ✅ Make informed decisions about model deployment configurations
- ✅ Apply inference concepts across different AI modalities

## Next Steps

After completing these notebooks, you'll be ready to:
- Deploy models in production with appropriate performance tuning
- Build real-time AI applications with predictable latency
- Evaluate and compare different models for your use case
- Progress to advanced topics like continuous batching and speculative decoding