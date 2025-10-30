r"""Small training runner for ai.train_model

Usage (PowerShell):
    $env:DEBUG='1'; & 'C:/Users/MagicBowl/AppData/Local/Programs/Python/Python313/python.exe' ./scripts/train_runner.py --parquets ./a.parquet ./b.parquet --batch-size 8 --epochs 1 --save-path best_model.pth

This script ensures the repo root is on sys.path so `from ai import train_model` works
and exposes a few basic CLI options for safe short runs.
"""
import argparse
import os
import sys
from multiprocessing import freeze_support


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai import train_model


def parse_args():
    p = argparse.ArgumentParser(description='Run a short training using ai.train_model')
    p.add_argument('--parquets', nargs='+', required=True, help='Paths to parquet files')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--save-path', type=str, default='best_model.pth')
    p.add_argument('--debug', action='store_true', help='Enable debug mode (num_workers=0)')
    p.add_argument('--use-focal', action='store_true', help='Use focal loss instead of cross-entropy')
    p.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    p.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    p.add_argument('--scheduler', choices=['plateau', 'step', 'none'], default='plateau', help='LR scheduler type')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--weight-decay', type=float, default=1e-4, help='Adam weight decay')
    return p.parse_args()


def main():
    freeze_support()
    args = parse_args()

    # Set DEBUG env var for train_model to pick up num_workers behavior
    if args.debug:
        os.environ['DEBUG'] = '1'

    # build dict mapping basename (without ext) to absolute path
    parquet_files = {}
    for p in args.parquets:
        absp = os.path.abspath(p)
        if not os.path.exists(absp):
            raise FileNotFoundError(f'Parquet not found: {absp}')
        key = os.path.splitext(os.path.basename(absp))[0]
        parquet_files[key] = absp

    print('Starting training with:', parquet_files)
    model, best_acc, label_map = train_model(
        parquet_files,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        patience=args.patience,
        scheduler_type=(None if args.scheduler == 'none' else args.scheduler),
        save_path=args.save_path,
    )

    # save model (train_model also saves best_model.pth during training)
    try:
        torch_save = getattr(sys.modules.get('torch'), 'save', None)
        if torch_save is None:
            # import torch lazily
            import torch as _torch
            _torch.save(model.state_dict(), args.save_path)
        else:
            import torch as _torch
            _torch.save(model.state_dict(), args.save_path)
        print(f'Model saved to {args.save_path}')
    except Exception as e:
        print('Warning: failed to save model:', e)

    print('Best val acc:', best_acc)
    print('Label map:', label_map)


if __name__ == '__main__':
    main()
