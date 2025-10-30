"""
Simple GPU smoke test: prints torch & CUDA info, moves a tensor to GPU (if available)
and performs a small op to confirm computation on the device.
"""
import torch

def main():
    print('torch version:', torch.__version__)
    print('torch.version.cuda:', getattr(torch.version, 'cuda', None))
    print('cuda available:', torch.cuda.is_available())
    try:
        print('cuda device count:', torch.cuda.device_count())
    except Exception as e:
        print('device_count error:', e)

    if torch.cuda.is_available():
        dev = torch.device('cuda')
        try:
            print('current device index:', torch.cuda.current_device())
            print('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception as e:
            print('device info error:', e)

        # create a random tensor and do a simple operation
        x = torch.randn(3, 3)
        print('tensor on cpu:', x.device)
        x = x.to(dev)
        print('tensor moved to:', x.device)
        y = (x * 2.0).sum()
        print('operation result (scalar):', y.item())
    else:
        print('CUDA not available — smoke test skipped GPU operations.')

if __name__ == '__main__':
    main()
