GPU setup and usage guide
=========================

This project supports training on GPU when you have a CUDA-enabled PyTorch installed.

Quick checks
------------

1. Check NVIDIA driver and GPU visibility:

```powershell
nvidia-smi
```

1. Check the Python interpreter's PyTorch CUDA support:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.device_count())"
```

If the output shows `cuda available: True` and a valid CUDA version (e.g. `12.8`), your interpreter is CUDA-capable.

Installing PyTorch with CUDA
----------------------------

Use the official PyTorch selector at [PyTorch Get Started](https://pytorch.org/get-started/locally/) to pick the right command for your OS and desired CUDA version. Example commands:

Pip (example for CUDA 12.8 / cu128):

```powershell
# remove cpu-only first (optional)
python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

Conda (example for CUDA 12.8):

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
```

Note: replace CUDA version with the one compatible with your NVIDIA driver (see `nvidia-smi` driver version). Prefer conda on Windows for fewer dependency issues.

Project-specific helpers
------------------------

- `ai.print_cuda_info()` — prints PyTorch and `nvidia-smi` info (already in `ai.py`).
- `scripts/gpu_smoketest.py` — small script that moves a tensor to GPU and performs a tiny op.

Run examples
------------

1. Smoke test (using the interpreter that has CUDA-enabled PyTorch):

```powershell
python scripts/gpu_smoketest.py
```

1. Train (example):

```python
from ai import train_model
train_model({'a': 'a.parquet', 'b': 'b.parquet'}, batch_size=32, num_epochs=10)
```

Performance notes
-----------------

- `ai.train_model` automatically uses GPU if available, sets `pin_memory=True` for DataLoader, and enables `cudnn.benchmark`.
- Mixed precision (AMP) is enabled automatically when CUDA is available to speed up training and reduce memory usage.
- For debugging in VS Code, use the provided `.vscode/launch.json` configuration `Python: Debug Current File` which sets `DEBUG=1` and uses `num_workers=0` to avoid multiprocessing issues.

If you want, I can add a small script to launch training with typical hyperparameters or integrate a CLI entry point.
