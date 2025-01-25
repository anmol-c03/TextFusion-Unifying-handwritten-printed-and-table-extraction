from transformers import AutoModelForObjectDetection
import torch
from PIL import Image


model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

file_path = '/Users/anmolchalise/Desktop/Major_project_final/Table_extraction/images/anmol.jpg'
image = Image.open(file_path).convert("RGB")
# let's display it a bit smaller
width, height = image.size
