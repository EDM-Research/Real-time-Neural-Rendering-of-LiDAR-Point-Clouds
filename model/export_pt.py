import torch
from model import UNet
import os

model = UNet(in_channels=5, out_channels=3, features=[64, 128, 256, 512])
checkpoint = torch.load("model.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval().cuda().half()

# Export to TorchScript
example_input = torch.randn(1, 5, 720, 960).cuda().half()  # Adjust to your input shape
traced_model = torch.jit.trace(model, example_input)
# directory the renderer will look for the model named model.pt
traced_model.save(os.getenv("HOME") + "/.render_cache/model.pt")
