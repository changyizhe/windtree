import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle
from model_set import *


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Load the trained model
model_save_path = "model1.pth"
model = SimpleCNN()
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Load and preprocess the image
image_path = "images_test/test1.jpg"
image = preprocess_image(image_path)

# Make a prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)

with open('class_def.pkl', 'rb') as file:
    class_def = pickle.load(file)

class_def_inv = {v: k for k, v in class_def.items()}

# Print the predicted class
print ("predicted class: "+class_def_inv[predicted.item()])
# print(f"Predicted class: {predicted.item()}")

