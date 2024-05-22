from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to(device)
img = 'Test Images\download.jpeg'
image = Image.open(img)

prompt = "Question: how many cats are there?\nAnswer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

outputs = model(**inputs)