# Releases

This folder contains exported model artifacts from training runs.

Files included

- `model_ts_long.pt` — TorchScript (traced) version of the trained model. Load with `torch.jit.load` and run on CPU or CUDA.
- `model_long.onnx` — ONNX export (opset 18). Run with `onnxruntime` or other ONNX runtimes.

Quick examples

1) TorchScript (Python)

```python
import torch
from PIL import Image
from torchvision import transforms

# adjust path as needed
ts_path = r"./releases/model_ts_long.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(ts_path, map_location=device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open('some_image.jpg').convert('RGB')
inp = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    out = model(inp)
    probs = torch.softmax(out, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0, pred].item()

print('pred', pred, 'conf', conf)
```

2) ONNX (Python + onnxruntime)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

sess = ort.InferenceSession(r"./releases/model_long.onnx")
input_name = sess.get_inputs()[0].name

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open('some_image.jpg').convert('RGB')
arr = transform(img).unsqueeze(0).numpy()
outputs = sess.run(None, {input_name: arr})
probs = torch.softmax(torch.from_numpy(outputs[0]), dim=1).numpy()
pred = int(np.argmax(probs, axis=1)[0])
conf = float(np.max(probs))
print('pred', pred, 'conf', conf)
```

Notes and troubleshooting

- If ONNX export or runtime fails, ensure `onnx`, `onnxscript` and `onnxruntime` are installed in your Python environment: `python -m pip install onnx onnxscript onnxruntime`.
- TorchScript is generally the easiest way to deploy PyTorch models for CPU/CUDA inference. ONNX is useful for cross-framework runtime interoperability.
- If you need a dynamic-batch ONNX model, exporting with `dynamic_axes` is possible but may require TorchDynamo adjustments; file `model_long.onnx` was exported with a fixed batch=1 to maximize compatibility.

If you want, I can also:

- Add small example scripts to `scripts/` for TorchScript and ONNX inference (`scripts/infer_torchscript.py`, `scripts/infer_onnx.py`).
- Package the releases into a ZIP for download.

## Model release directory

Contains exported TorchScript model for deployment.

Files:

- `model_ts.pt` — traced TorchScript file exported from `SimpleCNN`.

Notes:

- This is a CPU-traced TorchScript. You can load it with `torch.jit.load('model_ts.pt')` and move to GPU with `.to('cuda')` if desired.
- The file is a binary artifact; if you prefer not to track it in git, consider storing it in an artifact store or cloud bucket.
