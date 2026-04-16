import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# PATHS (NO CHANGE)
IMG_DIR = "DFUC2022_train_release/DFUC2022_train_images"
MASK_DIR = "DFUC2022_train_release/DFUC2022_train_masks"

# DATASET CLASS
class DFUDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (256, 256))
        mask = mask / 255.0

        mask = np.expand_dims(mask, axis=0)
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# SIMPLE U-NET
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU()
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        d = self.up(e2)
        d = torch.cat([d, e1], dim=1)
        d = self.dec(d)

        return torch.sigmoid(self.out(d))

# LOAD DATA
dataset = DFUDataset(IMG_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print("✅ Dataset loaded:", len(dataset))

# MODEL
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# TRAINING LOOP
for epoch in range(5):
    total_loss = 0
    print(f"\n🚀 Starting Epoch {epoch+1}/5")

    for batch_idx, (img, mask) in enumerate(loader):
        pred = model(img)
        loss = loss_fn(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # PRINT EVERY 20 BATCHES
        if batch_idx % 20 == 0:
            print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} Completed | Total Loss: {total_loss:.4f}")

# SAVE MODEL
torch.save(model.state_dict(), "unet_model.pth")

print("\n🎉 U-Net Training Completed Successfully!")