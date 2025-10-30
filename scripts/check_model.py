"""Check and report contents of a saved PyTorch model checkpoint (best_model.pth).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai import SimpleCNN
import torch


def main():
    path = os.path.join(ROOT, 'best_model.pth')
    if not os.path.exists(path):
        print('best_model.pth not found at', path)
        return
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and all(isinstance(k, str) for k in sd.keys()):
        print('state_dict keys count:', len(sd))
        print('Top keys (up to 10):', list(sd.keys())[:10])
        # try loading into model
        # If classifier output size differs, allow non-strict load
        model = SimpleCNN(num_classes=2)
        res = model.load_state_dict(sd, strict=False)
        print('load_state_dict result:', res)
        total_params = sum(p.numel() for p in model.parameters())
        print('total model parameters:', total_params)
    else:
        print('Loaded object is not a flat state_dict; type:', type(sd))


if __name__ == '__main__':
    main()
