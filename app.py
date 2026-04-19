import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# Load model
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

st.title("🐱 Cat vs Dog Classifier")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, 1).item()

    if pred == 0:
        st.success("🐱 Cat")
    else:
        st.success("🐶 Dog")