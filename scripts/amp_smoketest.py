"""AMP smoke test: constructs model, optimizer and runs one training step using torch.amp when CUDA is available."""
import os
import sys
import torch

# Ensure repo root is on sys.path so `from ai import ...` works when run from scripts/
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai import SimpleCNN


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device ->', device)
    model = SimpleCNN(num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # fake batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    y = torch.randint(0, 2, (batch_size,))

    x = x.to(device)
    y = y.to(device)

    use_amp = device.type == 'cuda'
    # GradScaler in this PyTorch build may not accept device_type kwarg; use default constructor
    scaler = torch.amp.GradScaler() if use_amp else None

    model.train()
    opt.zero_grad()
    if use_amp:
        with torch.amp.autocast(device_type='cuda'):
            out = model(x)
            loss = criterion(out, y)
        print('loss (amp):', loss.item())
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        out = model(x)
        loss = criterion(out, y)
        print('loss:', loss.item())
        loss.backward()
        opt.step()

    print('amp test done')


if __name__ == '__main__':
    main()
