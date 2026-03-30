import argparse
import random
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.models as models

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class FoodEmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        base = models.efficientnet_b0(weights="DEFAULT")
        self.backbone = base.features
        self.pool = base.avgpool

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        return F.normalize(x, dim=1)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class FolderTripletDataset(Dataset):
    def __init__(self, root_dir, transform, n_per_class=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.rng = random.Random(SEED)

        self.samples = []
        self.labels = []
        self.label_names = []

        exts = {".jpg", ".jpeg", ".png", ".webp"}

        for i, folder in enumerate(sorted(self.root_dir.iterdir())):
            if not folder.is_dir():
                continue

            imgs = [f for f in folder.iterdir() if f.suffix.lower() in exts]

            if n_per_class and len(imgs) > n_per_class:
                imgs = self.rng.sample(imgs, n_per_class)

            for img in imgs:
                self.samples.append(img)
                self.labels.append(i)

            self.label_names.append(folder.name)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor = self.samples[idx]
        label = self.labels[idx]

        pos_idx = np.random.choice(np.where(self.labels == label)[0])
        neg_idx = np.random.choice(np.where(self.labels != label)[0])

        def load(p):
            return self.transform(Image.open(p).convert("RGB"))

        return load(anchor), load(self.samples[pos_idx]), load(self.samples[neg_idx])

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = FolderTripletDataset(
        args.data_dir,
        transform=get_transform(),
        n_per_class=args.n_per_class
    )

    # split train/val
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch)

    model = FoodEmbeddingNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=1.0)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for a,p,n in train_loader:
            a,p,n = a.to(device), p.to(device), n.to(device)

            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for a,p,n in val_loader:
                a,p,n = a.to(device), p.to(device), n.to(device)
                loss = criterion(model(a), model(p), model(n))
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = Path(args.output)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                "model_state_dict": model.state_dict(),
                "menus": dataset.label_names,
                "menu_to_idx": {m:i for i,m in enumerate(dataset.label_names)},
                "config": {
                    "embed_dim": 128,
                    "backbone": "efficientnet_b0",
                    "image_size": 224
                }
            }, save_path)

            print("✅ Saved best model")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--n_per_class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output", default="checkpoints/food_classifier.pth")

    args = parser.parse_args()
    train(args)