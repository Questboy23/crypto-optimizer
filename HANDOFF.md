# HANDOFF (written 2026-10-23 03:06:52)
## GPU
GPU=none. 5 Kaggle pushes (script and notebook formats, T4 flag) all ran CPU-only (torch +cpu, no nvidia-smi).
Stored metadata (kaggle kernels pull -m): {"id":"reedostrov/gpu-gputest","id_no":140341034,"title":"gpu-gputest","code_file":"gpu-gputest.ipynb","language":"python","kernel_type":"notebook","is_private":true,"enable_gpu":true,"enable_tpu":false,"enable_internet":true,"keywords":[],"dataset_sources":["reedostrov/kaggle-gpu-project"],"kernel_sources":[],"competition_sources":[],"model_sources":[],"docker_image":"gcr.io/kaggle-images/python@sha256:17b2fc7f27c25c6c2fd4230ec8e66097ab2b89fc88e1e4d6f338b72e1ee0bc3e","machine_shape":"NvidiaTeslaT4"}
Kaggle stored enable_gpu=true and machine_shape=NvidiaTeslaT4, so no field was dropped; session still had no GPU (likely phone verification or GPU quota).
## Disk
Volume is 100% full (3 to 5 GB free). My partial Kraken download (2 GB part00) was deleted. Do not download the 9.5 GB bulk zip; read daily files via HTTP range requests.
