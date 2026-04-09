import torch

print("torch version:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.ones(3, device=device)
print(x)
print("device:", x.device)