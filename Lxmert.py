import torch
from transformers import LxmertTokenizer, LxmertForQuestionAnswering
from PIL import Image
from torchvision import transforms
import requests
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load the tokenizer and model
tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-vqa-uncased')
model = LxmertForQuestionAnswering.from_pretrained('unc-nlp/lxmert-vqa-uncased')

# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

image_path = 'Test Images\download.jpeg'
image = load_image(image_path)

# Load pre-trained Faster R-CNN model
detector = fasterrcnn_resnet50_fpn(pretrained=True)
detector.eval()

# Transform image for the detector
transform = transforms.Compose([
    transforms.Resize((800, 800)),  # Resize the image for better detection
    transforms.ToTensor()
])
image_tensor = transform(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

# Get object detection features
with torch.no_grad():
    outputs = detector(image_tensor)

# Extract bounding boxes
boxes = outputs[0]['boxes']

# Crop and resize regions from the original image
regions = []
for box in boxes:
    x1, y1, x2, y2 = box.tolist()
    region = image.crop((x1, y1, x2, y2))  # Crop the region from the original image
    region = region.resize((224, 224))  # Resize to the input size of LXMERT
    regions.append(region)

# Convert regions to tensor and stack them
regions_tensor = torch.stack([transforms.ToTensor()(region) for region in regions])

# Prepare visual embeddings
visual_embeds = regions_tensor.unsqueeze(0)  # Add batch dimension

# Prepare positional encodings for visual embeddings
num_boxes = visual_embeds.shape[1]
visual_pos = torch.arange(1, num_boxes + 1).unsqueeze(0).expand(visual_embeds.shape[:2]).to(visual_embeds.device)

# Prepare the question
question = "What is in the image?"
inputs = tokenizer(question, return_tensors="pt")

visual_pos = visual_pos.unsqueeze(-1).expand(-1, -1, visual_embeds.shape[2]).to(visual_embeds.device)

# Add visual embeddings and positional encodings to the inputs dictionary
inputs['visual_feats'] = visual_embeds
inputs['visual_pos'] = visual_pos
assert inputs['visual_feats'].shape[2] == 2048, "Incorrect visual embeddings shape"

# Forward pass to get the prediction
with torch.no_grad():
    outputs = model(**inputs)
    answer_logits = outputs.logits

# Get the predicted answer
predicted_answer = tokenizer.decode(answer_logits.argmax(-1).item())

print(f"Question: {question}")
print(f"Answer: {predicted_answer}")