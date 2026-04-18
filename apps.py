import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import io

app = FastAPI(title="Image Classification API")

# ----- Model Definition -----
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----- Load Model -----
# For demo, we'll initialize a fresh model.
# In a real scenario, load pre-trained model.
num_classes = 3  # e.g., cat, dog, car
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# ----- Image Transformations -----
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# ----- Class Labels -----
classes = ['cat', 'dog', 'car']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]

    return {"class": class_name}
