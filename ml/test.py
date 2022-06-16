import torch

def test_gpu():
    print(f"if cuda available: {torch.cuda.is_available()}")
    print(f"device num: {torch.cuda.device_count()}")
    print(f"current device id: {torch.cuda.current_device()}")
    print(f"current device name: {torch.cuda.get_device_name(0)}\n")


if __name__ == "__main__":
    test_gpu()