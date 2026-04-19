import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

# =========================
# LOAD & FIX CSV
# =========================
df = pd.read_csv("cat_vs_dog.csv")

# Fix filename issues
df["file_path"] = df["file_path"].astype(str)
df["file_path"] = df["file_path"].str.replace("_", ".", regex=False)
df["file_path"] = df["file_path"].str.replace("\\", "/", regex=False)

# Label encoding
label_map = {"cat": 0, "dog": 1}
df["label"] = df["label"].map(label_map)

print("✅ CSV Loaded")
print(df.head())

# =========================
# CUSTOM DATASET
# =========================
class CatDogDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_rel_path = self.df.iloc[idx]["file_path"]
        img_path = os.path.join(self.root_dir, img_rel_path)

        # Skip missing files safely
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping missing file: {img_path}")
            return self.__getitem__((idx + 1) % len(self.df))

        image = Image.open(img_path).convert("RGB")
        label = self.df.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================
# TRANSFORM (FAST)
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =========================
# DATA LOADER
# =========================
dataset = CatDogDataset(df, "images", transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================
# MODEL
# =========================
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

model = CNN()

# =========================
# TRAINING SETUP
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAIN MODEL
# =========================
print("🚀 Training started...")

for epoch in range(2):
    for i, (images, labels) in enumerate(loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "model.pth")

print("✅ Training completed & model saved as model.pth")