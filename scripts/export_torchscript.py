"""Export the trained SimpleCNN model to TorchScript and verify by running one sample prediction.

Usage:
  python scripts/export_torchscript.py --model best_model.pth --out model_ts.pt --parquet a.parquet --index 0
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai import SimpleCNN, val_transform, predict_image_bytes
import torch
import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.path.join(ROOT, 'best_model.pth'))
    p.add_argument('--out', default=os.path.join(ROOT, 'model_ts.pt'))
    p.add_argument('--parquet', default=os.path.join(ROOT, 'a.parquet'))
    p.add_argument('--index', type=int, default=0)
    return p.parse_args()


def load_image_bytes_from_parquet(parquet_path, index=0):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if len(df) == 0:
        raise RuntimeError('Parquet has no rows')
    row = df.iloc[index]
    img = row['image']
    if isinstance(img, dict):
        for k in ('bytes', 'data', 'buf', 'blob', 'value', 'values'):
            if k in img:
                return img[k]
        vals = list(img.values())
        if len(vals) == 1:
            return vals[0]
    return img


def main():
    args = parse_args()

    print('Loading model state_dict from', args.model)
    sd = torch.load(args.model, map_location='cpu')
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(sd)
    model.eval()

    # Trace on CPU with example input
    example = torch.randn(1, 3, 224, 224)
    print('Tracing model to TorchScript...')
    try:
        ts = torch.jit.trace(model, example)
        ts.save(args.out)
        print('Saved TorchScript to', args.out)
    except Exception as e:
        print('Failed to trace/save TorchScript:', e)
        raise

    # Verify by loading the TorchScript and running predict on a sample image
    print('Loading TorchScript back and running verification prediction...')
    ts2 = torch.jit.load(args.out, map_location='cpu')
    ts2.eval()

    img_bytes = load_image_bytes_from_parquet(args.parquet, args.index)
    if isinstance(img_bytes, memoryview):
        img_bytes = bytes(img_bytes)

    # predict_image_bytes expects a nn.Module, but TorchScript module supports same forward
    device = torch.device('cpu')
    pred, conf = predict_image_bytes(ts2, img_bytes, val_transform, device)
    print('Verification prediction -> pred:', pred, 'conf:', conf)


if __name__ == '__main__':
    main()
