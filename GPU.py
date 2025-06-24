import torch, torchvision, torchaudio
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
print("TorchVision:", torchvision.__version__)
print("Torchaudio:", torchaudio.__version__)