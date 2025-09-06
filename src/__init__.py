import torch


INDEX_NAME = "medbot"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
