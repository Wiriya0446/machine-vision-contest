"""
Food Image Pair Selector — แบบที่ 1: เทรนแยกตามเมนู (5 Models)
================================================================
การเปลี่ยนแปลงจากเวอร์ชันก่อน:
  1. รองรับ csv_path_3 / image_dir_3
  2. ใช้ Optuna หา hyperparameters ที่ดีที่สุดอัตโนมัติ
  3. image_dir รองรับทั้งแบบรูปรวมโฟลเดอร์เดียว และแบบแยกโฟลเดอร์ย่อยตามเมนู
  4. GPU optimization: pin_memory, AMP (mixed precision), cudnn.benchmark
"""

import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # ── แหล่งข้อมูลที่ 1 ──
    "csv_path":  "C:\\Users\\ASUS\\Documents\\machine vision\\contest\\data_from_questionaire.csv",
    "image_dir": "C:\\Users\\ASUS\\Documents\\machine vision\\contest\\Questionair Images",

    # ── แหล่งข้อมูลที่ 2 (ใส่ None เพื่อปิด) ──
    "csv_path_2":  "C:\\Users\\ASUS\\Documents\\machine vision\\contest\\data_from_intragram_2.csv",
    "image_dir_2": "C:\\Users\\ASUS\\Documents\\machine vision\\contest\\Intragram Images [Original]",

    # ── แหล่งข้อมูลที่ 3 (ใส่ None เพื่อปิด) ──
    "csv_path_3":  "C:\\Users\\ASUS\\Documents\\machine vision\\contest\\auto_labeled_filtered.csv",   # ← ใส่ path CSV ที่ 3 ตรงนี้
    "image_dir_3": None,   # ← ใส่ path โฟลเดอร์รูปที่ 3 ตรงนี้

    # ── Model ──
    "model_name": "efficientnet_b0",
    "image_size": 224,
    "pretrained": True,

    # ── Training (ค่า default ถ้าปิด Optuna) ──
    "batch_size":    32,
    "epochs":        30,
    "learning_rate": 3e-4,
    "weight_decay":  3e-4,
    "dropout":       0.3,

    # ── Optuna ──
    "use_optuna":    True,  # False = ใช้ค่า default ข้างบนเลย
    "optuna_trials": 20,    # จำนวนครั้งที่ลอง (เพิ่มได้ถ้าต้องการละเอียดขึ้น)
    "optuna_epochs": 15,    # epochs ต่อ trial (น้อยกว่าเทรนจริงเพื่อความเร็ว)

    # ── Data split ──
    "val_split":   0.15,
    "test_split":  0.15,
    "random_seed": 42,

    # ── Augmentation ──
    "use_augmentation": True,

    # ── Output ──
    "save_dir": "checkpoints_per_menu",

    # ── Early stopping ──
    "patience": 15,

    # ── เมนูที่จะเทรน (None = ทั้งหมด) ──
    "menus_to_train": None,

    # ── Source weights — น้ำหนักของแต่ละแหล่งข้อมูล ──
    # human label (csv 1,2) น่าเชื่อถือกว่า auto label (csv 3)
    # ตัวเลขยิ่งสูง = สุ่มเลือกบ่อยกว่าระหว่าง train
    "source_weights": {
        1: 3.0,   # data_from_questionaire.csv  (human label คุณภาพสูงสุด)
        2: 3.0,   # data_from_intragram_2.csv   (human label)
        3: 1.0,   # auto_labeled_filtered.csv   (auto label)
    },

    # ── GPU optimization ──
    # num_workers=0 ปลอดภัยบน Windows, ลอง 2-4 ได้ถ้าไม่มี error
    "num_workers": 2,
    "use_amp":     True,   # Automatic Mixed Precision เร็วขึ้น ~30-50% บน RTX
}

# ── ชื่อ subfolder ของแต่ละเมนู ──
# ใช้เมื่อ image_dir แยกโฟลเดอร์ย่อยตามเมนู เช่น image_dir/Sushi/001.jpg
MENU_SUBFOLDERS = {
    "Sushi":   "Sushi",
    "Ramen":   "Ramen",
    "Pizza":   "Pizza",
    "Burger":  "Burger",
    "Dessert": "Dessert",
}


# ─────────────────────────────────────────────
#  IMAGE INDEX — scan ครั้งเดียวตอนเริ่ม แล้ว lookup O(1)
# ─────────────────────────────────────────────
def build_image_index(image_dirs):
    """
    Scan ทุกโฟลเดอร์ใน image_dirs ครั้งเดียวตอนเริ่มโปรแกรม
    สร้าง dict {filename: Path} เพื่อ lookup เร็วตอนโหลดรูป
    ถ้าชื่อซ้ำกันข้ามโฟลเดอร์ จะใช้อันแรกที่เจอ (image_dir ก่อน image_dir_2 ก่อน image_dir_3)
    """
    exts  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    index = {}
    for d in image_dirs:
        d = Path(d)
        if not d.exists():
            print(f"  [WARNING] ไม่พบโฟลเดอร์: {d}")
            continue
        print(f"  Scanning {d} ...", end=" ", flush=True)
        count = 0
        for f in d.rglob("*"):
            if f.suffix.lower() in exts and f.name not in index:
                index[f.name] = f
                count += 1
        print(f"{count:,} รูป")
    print(f"  รวม index ทั้งหมด: {len(index):,} รูป")
    return index


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class FoodPairDataset(Dataset):
    def __init__(self, df, image_index, transform=None):
        """
        image_index : dict {filename: Path} จาก build_image_index()
        """
        self.df          = df.reset_index(drop=True)
        self.image_index = image_index
        self.transform   = transform

    def __len__(self):
        return len(self.df)

    def _load_image(self, filename):
        p = self.image_index.get(filename)
        if p is not None:
            return Image.open(p).convert("RGB")
        print(f"  [WARNING] ไม่พบรูป: {filename}")
        return Image.new("RGB", (CONFIG["image_size"], CONFIG["image_size"]))

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img1     = self._load_image(row["Image 1"])
        img2     = self._load_image(row["Image 2"])
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        combined = torch.cat([img1, img2], dim=0)
        label    = int(row["Winner"]) - 1
        return combined, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
class FoodSelectorNet(nn.Module):
    def __init__(self, model_name="efficientnet_b0", dropout=0.5, pretrained=True):
        super().__init__()
        model_map = {
            "efficientnet_b0": (models.efficientnet_b0, 1280),
            "efficientnet_b1": (models.efficientnet_b1, 1280),
            "efficientnet_b2": (models.efficientnet_b2, 1408),
            "efficientnet_b3": (models.efficientnet_b3, 1536),
        }
        builder, feat_dim = model_map.get(model_name, (models.efficientnet_b0, 1280))
        base = builder(weights="IMAGENET1K_V1" if pretrained else None)

        old = base.features[0][0]
        new = nn.Conv2d(6, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight
        base.features[0][0] = new

        self.backbone   = base.features
        self.pool       = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ─────────────────────────────────────────────
#  TRANSFORMS & NORMALIZE
# ─────────────────────────────────────────────
def get_transforms(mode="train"):
    size = CONFIG["image_size"]
    if mode == "train" and CONFIG["use_augmentation"]:
        return transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


class NormalizeSixChannel:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406])
        self.std  = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225])

    def __call__(self, x):
        return (x - self.mean[:, None, None]) / self.std[:, None, None]


_normalizer = NormalizeSixChannel()


def collate_fn(batch):
    imgs, labels = zip(*batch)
    return _normalizer(torch.stack(imgs)), torch.stack(labels)


# ─────────────────────────────────────────────
#  WEIGHTED SAMPLER — สุ่มข้อมูลตาม source weight
# ─────────────────────────────────────────────
def make_weighted_sampler(df):
    """
    สร้าง WeightedRandomSampler จาก _source column
    แถวที่มาจาก source น้ำหนักสูงจะถูกสุ่มเลือกบ่อยกว่า
    """
    from torch.utils.data import WeightedRandomSampler
    weight_map = CONFIG.get("source_weights", {1: 1.0, 2: 1.0, 3: 1.0})
    # ถ้าไม่มี _source column ให้ทุกแถวมีน้ำหนักเท่ากัน
    if "_source" not in df.columns:
        weights = [1.0] * len(df)
    else:
        weights = [weight_map.get(int(s), 1.0) for s in df["_source"]]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


# ─────────────────────────────────────────────
#  BUILD DATALOADERS
# ─────────────────────────────────────────────
def build_loaders(train_df, val_df, test_df, image_index, batch_size):
    pin = torch.cuda.is_available()
    nw  = CONFIG["num_workers"]
    pw  = nw > 0

    kw = dict(num_workers=nw, pin_memory=pin,
              collate_fn=collate_fn, persistent_workers=pw)

    sampler = make_weighted_sampler(train_df)
    train_loader = DataLoader(
        FoodPairDataset(train_df, image_index, get_transforms("train")),
        batch_size=batch_size, sampler=sampler,
        num_workers=kw["num_workers"], pin_memory=kw["pin_memory"],
        collate_fn=kw["collate_fn"], persistent_workers=kw["persistent_workers"])
    val_loader = DataLoader(
        FoodPairDataset(val_df, image_index, get_transforms("val")),
        batch_size=batch_size * 2, shuffle=False, **kw)
    test_loader = DataLoader(
        FoodPairDataset(test_df, image_index, get_transforms("val")),
        batch_size=batch_size * 2, shuffle=False, **kw)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
#  TRAIN / EVALUATE (รองรับ AMP)
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    use_amp = CONFIG["use_amp"] and device.type == "cuda"

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    use_amp = CONFIG["use_amp"] and device.type == "cuda"

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
#  CORE TRAINING LOOP
# ─────────────────────────────────────────────
def run_training(menu, train_loader, val_loader, device,
                 lr, weight_decay, dropout, epochs, patience,
                 model_path=None):
    model     = FoodSelectorNet(CONFIG["model_name"], dropout, CONFIG["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=(CONFIG["use_amp"] and device.type == "cuda"))

    best_val_acc     = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta       = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl, va, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        # แสดง log เฉพาะตอน final training (model_path ถูกกำหนด)
        if model_path:
            mark = " ✓" if va > best_val_acc else ""
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss {tl:.4f}/{vl:.4f} | Acc {ta:.4f}/{va:.4f} | "
                  f"{time.time()-t0:.1f}s{mark}")

        if va > best_val_acc:
            best_val_acc     = va
            patience_counter = 0
            if model_path:
                torch.save({
                    "menu":             menu,
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          va,
                    "config":           CONFIG,
                }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if model_path:
                    print(f"  Early stopping epoch {epoch}")
                break

    return best_val_acc, history, model


# ─────────────────────────────────────────────
#  OPTUNA SEARCH
# ─────────────────────────────────────────────
def optuna_search(menu, train_df, val_df, image_index, device):
    print(f"\n  Optuna Search: {menu} "
          f"({CONFIG['optuna_trials']} trials x {CONFIG['optuna_epochs']} epochs)")

    def objective(trial):
        lr           = trial.suggest_float("lr",           1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        dropout      = trial.suggest_float("dropout",      0.2,  0.6)
        batch_size   = trial.suggest_categorical("batch_size", [16, 32])

        t_loader, v_loader, _ = build_loaders(
            train_df, val_df, val_df, image_index, batch_size
        )
        val_acc, _, _ = run_training(
            menu, t_loader, v_loader, device,
            lr=lr, weight_decay=weight_decay, dropout=dropout,
            epochs=CONFIG["optuna_epochs"], patience=5,
            model_path=None
        )
        return val_acc

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=CONFIG["random_seed"])
    )
    study.optimize(objective, n_trials=CONFIG["optuna_trials"],
                   show_progress_bar=True)

    best = study.best_params
    print(f"  Best params: lr={best['lr']:.2e} | wd={best['weight_decay']:.2e} | "
          f"dropout={best['dropout']:.2f} | batch={best['batch_size']} | "
          f"val_acc={study.best_value*100:.1f}%")
    return best


# ─────────────────────────────────────────────
#  TRAIN ONE MENU
# ─────────────────────────────────────────────
def train_one_menu(menu, df_menu, image_index, device, save_dir):
    print(f"\n{'='*60}")
    print(f"  เทรน Model: {menu} ({len(df_menu):,} คู่)")
    print(f"{'='*60}")

    # Split
    train_val, test_df = train_test_split(
        df_menu, test_size=CONFIG["test_split"],
        random_state=CONFIG["random_seed"])
    val_ratio = CONFIG["val_split"] / (1 - CONFIG["test_split"])
    train_df, val_df = train_test_split(
        train_val, test_size=val_ratio,
        random_state=CONFIG["random_seed"])

    print(f"  Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

    # ── Optuna ──
    if CONFIG["use_optuna"]:
        best_params  = optuna_search(menu, train_df, val_df, image_index, device)
        lr           = best_params["lr"]
        weight_decay = best_params["weight_decay"]
        dropout      = best_params["dropout"]
        batch_size   = best_params["batch_size"]
    else:
        lr           = CONFIG["learning_rate"]
        weight_decay = CONFIG["weight_decay"]
        dropout      = CONFIG["dropout"]
        batch_size   = CONFIG["batch_size"]

    # ── Final Training ──
    print(f"\n  Final Training ({CONFIG['epochs']} epochs)")
    print(f"  lr={lr:.2e} | wd={weight_decay:.2e} | dropout={dropout:.2f} | batch={batch_size}")

    train_loader, val_loader, test_loader = build_loaders(
        train_df, val_df, test_df, image_index, batch_size
    )

    model_path = save_dir / f"model_{menu.lower()}.pth"
    t0 = time.time()

    best_val_acc, history, model = run_training(
        menu, train_loader, val_loader, device,
        lr=lr, weight_decay=weight_decay, dropout=dropout,
        epochs=CONFIG["epochs"], patience=CONFIG["patience"],
        model_path=model_path
    )

    print(f"  เวลาเทรน: {(time.time()-t0)/60:.1f} นาที")

    # ── Test ──
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n  [{menu}] Test Acc: {test_acc*100:.1f}% | Best Val: {best_val_acc*100:.1f}%")
    print(classification_report(labels, preds,
                                target_names=["รูปที่ 1 ชนะ", "รูปที่ 2 ชนะ"],
                                zero_division=0))

    _save_plot(history, save_dir / f"history_{menu.lower()}.png", menu)
    return test_acc, best_val_acc


def _save_plot(history, path, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    e = range(1, len(history["train_loss"]) + 1)
    ax1.plot(e, history["train_loss"], label="Train", color="#2196F3")
    ax1.plot(e, history["val_loss"],   label="Val",   color="#F44336")
    ax1.set_title(f"Loss — {title}"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(e, history["train_acc"], label="Train", color="#2196F3")
    ax2.plot(e, history["val_acc"],   label="Val",   color="#F44336")
    ax2.set_title(f"Accuracy — {title}"); ax2.legend()
    ax2.set_ylim(0, 1); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Food Selector — Per-Menu Training (5 Models)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  AMP    : {'เปิด' if CONFIG['use_amp'] else 'ปิด'}")
        torch.backends.cudnn.benchmark = True   # เร็วขึ้นเมื่อ input size คงที่

    save_dir = Path(CONFIG["save_dir"])
    save_dir.mkdir(exist_ok=True)

    # ── โหลดและรวม CSV ทุกแหล่ง ──
    dfs = []
    for i, (csv_key, dir_key) in enumerate([
        ("csv_path",   "image_dir"),
        ("csv_path_2", "image_dir_2"),
        ("csv_path_3", "image_dir_3"),
    ], start=1):
        csv_p = CONFIG.get(csv_key)
        if not csv_p:
            continue
        print(f"\n  แหล่งข้อมูลที่ {i}: {csv_p}")
        tmp = pd.read_csv(csv_p)
        tmp.columns = tmp.columns.str.strip()
        tmp["_source"] = i   # เก็บ source id เพื่อใช้ทำ weighted sampling
        dfs.append(tmp)
        print(f"  {len(tmp):,} คู่")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  รวมทั้งหมด : {len(df):,} คู่")

    # ── รวม image_dirs ──
    image_dirs = [d for d in [
        CONFIG.get("image_dir"),
        CONFIG.get("image_dir_2"),
        CONFIG.get("image_dir_3"),
    ] if d]

    menus = CONFIG["menus_to_train"] or sorted(df["Menu"].unique())
    print(f"  เมนูที่จะเทรน  : {menus}")
    print(f"  Optuna        : {'เปิด (' + str(CONFIG['optuna_trials']) + ' trials)' if CONFIG['use_optuna'] else 'ปิด'}")

    # ── Build image index (scan ครั้งเดียว) ──
    print(f"\n  Building image index...")
    image_index = build_image_index(image_dirs)

    # ── เทรนแต่ละเมนู ──
    results = {}
    for menu in menus:
        df_menu = df[df["Menu"] == menu].copy()
        if len(df_menu) < 20:
            print(f"\n  {menu} มีข้อมูลน้อยเกินไป ({len(df_menu)} คู่) — ข้ามไป")
            continue
        test_acc, val_acc = train_one_menu(menu, df_menu, image_index, device, save_dir)
        results[menu] = {"test_acc": test_acc, "best_val_acc": val_acc}

    # ── สรุปผล ──
    print(f"\n{'='*60}")
    print(f"  สรุปผลทุกเมนู")
    print(f"{'='*60}")
    print(f"  {'เมนู':<12} {'Val Acc':>10} {'Test Acc':>10}")
    print(f"  {'-'*35}")
    for menu, r in results.items():
        print(f"  {menu:<12} {r['best_val_acc']*100:>9.1f}% {r['test_acc']*100:>9.1f}%")

    avg_test = np.mean([r["test_acc"] for r in results.values()])
    print(f"\n  Average Test Accuracy : {avg_test*100:.1f}%")
    print(f"  Model บันทึกที่: {save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()