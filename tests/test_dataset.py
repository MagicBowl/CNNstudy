import io
import base64
import os
import sys
import numpy as np
from PIL import Image
import torch

# Ensure workspace root is on sys.path so tests can import ai.py
root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from ai import ParquetImageDataset, train_transform


def make_jpeg_bytes(color=(255, 0, 0), size=(10, 10)):
    img = Image.new('RGB', size, color=color)
    b = io.BytesIO()
    img.save(b, format='JPEG')
    return b.getvalue()


def test_parquet_dataset_various_formats():
    img_bytes = make_jpeg_bytes()

    # Build fake dataset instance without calling __init__
    ds = ParquetImageDataset.__new__(ParquetImageDataset)
    ds.transform = train_transform
    ds.data = [
        img_bytes,  # raw bytes
        list(img_bytes),  # list of ints
        np.frombuffer(img_bytes, dtype=np.uint8),  # ndarray
        {'image': img_bytes},  # dict with key
        base64.b64encode(img_bytes).decode('ascii'),  # base64 string
    ]
    ds.labels = [0] * len(ds.data)

    for i in range(len(ds.data)):
        img_tensor, label = ds[i]
        # Tensor should be torch tensor shape [3, H, W]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.dim() == 3
        assert img_tensor.size(0) == 3
        assert label == 0


if __name__ == '__main__':
    # quick local run
    test_parquet_dataset_various_formats()
    print('tests passed')
