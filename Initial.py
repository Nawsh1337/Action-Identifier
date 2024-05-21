
import lib
from PIL import Image
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
import torch
from torchvision import transforms

def load_and_preprocess_image(img):
    image = Image.open(img).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image.unsqueeze(0)

img = 'Test Images\download.jpeg'
text = 'What is in the picture?'

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

inputs = tokenizer(text, return_tensors="pt")
visual_embeds = load_and_preprocess_image(img).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

# with torch.no_grad():
#     outputs = model(input_ids=inputs.input_ids, visual_embeds=visual_embeds)
#     answer_logits = outputs.logits

# # Get the predicted answer
# predicted_answer = tokenizer.decode(answer_logits.argmax(-1).item())

# print(f"Question: {text}")
# print(f"Answer: {predicted_answer}")
labels = torch.tensor([[0.0, 1.0]]).unsqueeze(0)  # Batch size 1, Num labels 2

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits

print(outputs)