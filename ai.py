import os
import io
import json
import base64
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


"""ai.py

Rewritten to match Parquet format where each row contains:
 - 'image': a dict that contains key 'bytes' with raw PNG/JPEG bytes
 - 'label': a string label such as 'oral_normal' or 'oral_scc'

This file provides:
 - ParquetImageDataset: robust loader + label mapping
 - train_transform / val_transform
 - SimpleCNN: small CNN with adaptive pooling
 - train_model(parquet_files, ...)
 - predict_image_bytes(model, image_bytes, transform, device)

Note: This module does NOT start training automatically when imported. Use
train_model(...) from your script or interactive session.
"""


print("当前工作目录是:", os.getcwd())


def print_cuda_info() -> Dict:
    """Print and return a dict with PyTorch/CUDA diagnostic info.

    Attempts to run `nvidia-smi` if available and includes its output.
    """
    info: Dict = {}
    try:
        info['torch_version'] = getattr(torch, '__version__', None)
        info['cuda_available'] = torch.cuda.is_available()
        info['cuda_version'] = getattr(torch.version, 'cuda', None)
        try:
            info['device_count'] = torch.cuda.device_count()
            info['devices'] = []
            for i in range(info['device_count']):
                try:
                    info['devices'].append(torch.cuda.get_device_name(i))
                except Exception:
                    info['devices'].append(f'device_{i}')
        except Exception:
            info['device_count'] = 0
            info['devices'] = []
    except Exception as e:
        info['torch_error'] = str(e)

    # try nvidia-smi
    try:
        import shutil, subprocess
        if shutil.which('nvidia-smi'):
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], stderr=subprocess.STDOUT, timeout=5)
            info['nvidia_smi'] = out.decode('utf-8').strip()
        else:
            info['nvidia_smi'] = None
    except Exception as e:
        info['nvidia_smi_error'] = str(e)

    # pretty print
    print('=== CUDA / PyTorch info ===')
    for k, v in info.items():
        print(f'{k}: {v}')
    print('===========================')
    return info


class ParquetImageDataset(Dataset):
    """Load images and labels from one or more Parquet files.

    Expected schema per file: columns 'image' and 'label'.
    - image: dict containing key 'bytes' (raw image bytes) or raw bytes directly
    - label: string label
    """

    def __init__(self, parquet_files: Dict[str, str], transform=None, label_map: Dict[str, int] = None):
        self.transform = transform
        self.data: List = []
        self.labels: List = []

        # Read and concatenate rows from provided parquet files
        for name, path in parquet_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f'Parquet file not found: {path}')
            df = pq.read_table(path).to_pandas()
            if 'image' not in df.columns or 'label' not in df.columns:
                raise ValueError(f'Parquet {path} must contain columns "image" and "label"')
            for _, row in df.iterrows():
                self.data.append(row['image'])
                self.labels.append(str(row['label']))

        # build label map
        unique = sorted(set(self.labels))
        if label_map is None:
            self.label2idx = {lab: i for i, lab in enumerate(unique)}
        else:
            self.label2idx = dict(label_map)
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        # convert labels to ints
        mapped = []
        for lab in self.labels:
            if lab in self.label2idx:
                mapped.append(self.label2idx[lab])
            else:
                nid = len(self.label2idx)
                self.label2idx[lab] = nid
                self.idx2label[nid] = lab
                mapped.append(nid)
        self.labels = mapped

        # try save label map
        try:
            with open('label_map.json', 'w', encoding='utf-8') as f:
                json.dump(self.label2idx, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw = self.data[idx]

        # extract bytes from dict if necessary
        image_bytes = raw
        try:
            if isinstance(raw, dict):
                # common keys that might contain the raw bytes
                for k in ('bytes', 'data', 'buf', 'blob', 'value', 'values'):
                    if k in raw:
                        image_bytes = raw[k]
                        break
                else:
                    vals = list(raw.values())
                    if len(vals) == 1:
                        image_bytes = vals[0]

            # numpy array -> bytes
            if isinstance(image_bytes, np.ndarray):
                image_bytes = image_bytes.tobytes()

            # list/tuple -> bytes
            elif isinstance(image_bytes, (list, tuple)):
                image_bytes = bytes(image_bytes)

            # string -> base64 or file path
            elif isinstance(image_bytes, str):
                try:
                    decoded = base64.b64decode(image_bytes)
                    if decoded:
                        image_bytes = decoded
                except Exception:
                    if os.path.exists(image_bytes):
                        with open(image_bytes, 'rb') as f:
                            image_bytes = f.read()

            if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
                raise TypeError(f'Unsupported image data type: {type(image_bytes)}')

            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise RuntimeError(f'Failed to load image at index {idx}: {e} (original type: {type(raw)})') from e

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# transforms exported for tests / external usage
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_model(parquet_files: Dict[str, str], batch_size: int = 32, num_epochs: int = 10,
                lr: float = 1e-3, weight_decay: float = 1e-4,
                use_focal: bool = False, focal_gamma: float = 2.0,
                patience: int = 5, scheduler_type: str = 'plateau',
                save_path: str = 'best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # If using CUDA, enable some recommended settings
    if device.type == 'cuda':
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass


    full_dataset = ParquetImageDataset(parquet_files, transform=train_transform)

    # split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    debug_mode = os.getenv('DEBUG', '0') == '1'
    num_workers = 0 if debug_mode else 4

    # pin_memory speeds up host->GPU transfers when using CUDA
    pin_memory = True if device.type == 'cuda' else False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # compute class weights to mitigate imbalance
    counts = {}
    for v in full_dataset.labels:
        counts[v] = counts.get(v, 0) + 1
    # produce weights inverse to frequency
    total_count = sum(counts.values())
    class_weights = [0.0] * len(full_dataset.label2idx)
    for idx, cnt in counts.items():
        class_weights[idx] = total_count / (len(counts) * cnt)

    model = SimpleCNN(num_classes=len(full_dataset.label2idx)).to(device)

    # Loss: either CrossEntropy with class weights or optional FocalLoss
    if use_focal:
        class FocalLoss(nn.Module):
            def __init__(self, gamma: float = 2.0, weight=None, reduction='mean'):
                super().__init__()
                self.gamma = gamma
                self.weight = weight
                self.reduction = reduction

            def forward(self, inputs, targets):
                ce = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
                pt = torch.exp(-ce)
                loss = ((1 - pt) ** self.gamma) * ce
                if self.reduction == 'mean':
                    return loss.mean()
                elif self.reduction == 'sum':
                    return loss.sum()
                return loss

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = FocalLoss(gamma=focal_gamma, weight=weight_tensor)
    else:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler: ReduceLROnPlateau (monitor val_loss) or StepLR as fallback
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    use_amp = device.type == 'cuda'
    # use new namespaced AMP API; avoid device_type kwarg for compatibility
    scaler = torch.amp.GradScaler() if use_amp else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        model.eval()
        # validation: compute val loss and accuracy
        correct = 0
        total = 0
        val_loss_accum = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(images)
                        vloss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    vloss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                # vloss may be a tensor
                val_loss_accum += vloss.item() if isinstance(vloss, torch.Tensor) else float(vloss)

        val_loss = val_loss_accum / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = 100.0 * correct / total if total > 0 else 0.0
        train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f'Epoch {epoch+1}/{num_epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}%')

        # scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # early stopping based on val_acc (and track val_loss for scheduler)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0
            try:
                torch.save(model.state_dict(), save_path)
            except Exception:
                torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'No improvement for {epochs_no_improve} epochs (patience={patience}). Stopping early.')
            break

    return model, best_val_acc, full_dataset.label2idx


def predict_image_bytes(model: nn.Module, image_bytes: bytes, transform, device):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outs = model(tensor)
        probs = torch.softmax(outs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()
    return pred, conf


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('ai.py 已加载为脚本。要训练模型，请调用 train_model({"a": "path/to/a.parquet", "b": "path/to/b.parquet"})')

