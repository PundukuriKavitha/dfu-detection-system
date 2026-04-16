import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# DEVICE
device = torch.device("cpu")

# PATHS
train_dir = "dataset/train"
val_dir = "dataset/valid"

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# DATASET
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=4, num_workers=0)

print("Classes:", train_data.classes)

# MODEL (EfficientNet)
model = models.efficientnet_b0(pretrained=True)

# MODIFY FINAL LAYER (5 classes: normal + grade1-4)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

model = model.to(device)

# LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TRAINING LOOP
for epoch in range(5):
    model.train()
    total_loss = 0

    print(f"\n🚀 Epoch {epoch+1}/5")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # PRINT PROGRESS
        if i % 10 == 0:
            print(f"Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} Completed | Total Loss: {total_loss:.4f}")

# SAVE MODEL
torch.save(model.state_dict(), "classifier.pth")

print("\n🎉 Classification Training Completed Successfully!")