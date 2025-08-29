import torch
from model import UNet
import torch_tensorrt
import os

H = 1440
W = 1440

model = UNet(in_channels=5, out_channels=3, features=[64, 128, 256, 512])
checkpoint = torch.load("model.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval().cuda().half()

example_input = [torch_tensorrt.Input(shape=(1,5,H,W), dtype=torch.half)]  # Adjust to your input shape

trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=example_input, enabled_precisions={torch.half})
torch_tensorrt.save(trt_gm, os.getenv("HOME") + "/.pcl_cache/trt_" + str(W) + ".ts", output_format="torchscript", inputs=[torch.randn(1,5,H,W, dtype=torch.half).cuda()])
