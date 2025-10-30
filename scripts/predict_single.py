"""Load saved model and run a single prediction on a row of a given parquet file.

Saves the image to disk with prediction metadata in the filename.

Usage:
    & 'C:/Users/MagicBowl/AppData/Local/Programs/Python/Python313/python.exe' ./scripts/predict_single.py --parquet ./a.parquet --index 0 --save-dir ./preds
"""
import os
import sys
import argparse
import io

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai import predict_image_bytes, ParquetImageDataset, val_transform
import torch
import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', required=True, help='Parquet file path to read one sample from')
    p.add_argument('--model', default=os.path.join(ROOT, 'best_model.pth'), help='Path to model state_dict')
    p.add_argument('--index', type=int, default=0, help='Row index in parquet to predict')
    p.add_argument('--save-dir', default=os.path.join(ROOT, 'preds'), help='Directory to save predicted image')
    return p.parse_args()


def load_image_bytes_by_index(parquet_path, index):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if df.shape[0] == 0:
        raise RuntimeError('Parquet has no rows')
    if index < 0 or index >= len(df):
        raise IndexError('Index out of range')
    row = df.iloc[index]
    img_field = row['image']
    # mimic dataset's extraction logic for simple cases
    image_bytes = img_field
    if isinstance(img_field, dict):
        for k in ('bytes', 'data', 'buf', 'blob', 'value', 'values'):
            if k in img_field:
                image_bytes = img_field[k]
                break
        else:
            vals = list(img_field.values())
            if len(vals) == 1:
                image_bytes = vals[0]

    # numpy array or memoryview -> bytes
    if isinstance(image_bytes, memoryview):
        image_bytes = bytes(image_bytes)
    return image_bytes, row.get('label')


def main():
    args = parse_args()
    parquet = os.path.abspath(args.parquet)
    model_path = os.path.abspath(args.model)
    print('parquet:', parquet)
    print('model:', model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device ->', device)

    # build model and load weights
    from ai import SimpleCNN
    model = SimpleCNN(num_classes=2).to(device)
    sd = torch.load(model_path, map_location='cpu')
    model.load_state_dict(sd)

    img_bytes, true_label = load_image_bytes_by_index(parquet, args.index)
    if isinstance(img_bytes, memoryview):
        img_bytes = bytes(img_bytes)
    # predict
    pred, conf = predict_image_bytes(model, img_bytes, val_transform, device)

    # load label map to map index -> name
    label_map_path = os.path.join(ROOT, 'label_map.json')
    label_name = None
    try:
        import json
        if os.path.exists(label_map_path):
            lm = json.load(open(label_map_path, 'r', encoding='utf-8'))
            inv = {v: k for k, v in lm.items()}
            label_name = inv.get(pred, None)
    except Exception:
        label_name = None

    # save image to disk for inspection
    os.makedirs(args.save_dir, exist_ok=True)
    out_name = f'pred_idx{args.index}_pred{pred}_conf{conf:.3f}_true_{(true_label or "unknown")}.jpg'
    out_path = os.path.join(args.save_dir, out_name)
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        im.save(out_path)
        print('Saved image to', out_path)
    except Exception as e:
        print('Failed to save image:', e)

    print('predicted index:', pred)
    if label_name:
        print('predicted label name:', label_name)
    print('confidence:', conf)
    print('true label (raw):', true_label)


if __name__ == '__main__':
    main()
