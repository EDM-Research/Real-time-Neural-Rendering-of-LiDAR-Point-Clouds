import torch
from model import UNet
import os

model = UNet(in_channels=5, out_channels=3, features=[64, 128, 256, 512])
checkpoint = torch.load("model.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval().half().cuda()

# Export to TorchScript
example_input = torch.randn(1, 5, 1440, 1920).half().cuda()  # Adjust to your input shape
traced_model = torch.jit.trace(model, example_input)
traced_model.save(os.getenv("HOME") + "/.pcl_cache/model.pt")
