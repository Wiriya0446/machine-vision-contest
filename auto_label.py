"""
Auto Label Script — Food Image Pairs (Batch + Checkpoint)
==========================================================
ใช้ model จาก train_per_menu.py (5 models แยกตามเมนู)
รองรับรูปจำนวนมาก (4,000-9,000+ รูป/เมนู)

Features:
  - Batch prediction → GPU ทำงานเต็มที่ เร็วขึ้น 10-20x
  - Checkpoint → ถ้า crash สามารถรันต่อจากจุดที่ค้างได้
  - บันทึก CSV ทุกครั้งที่เสร็จแต่ละเมนู กันข้อมูลหาย

โครงสร้างโฟลเดอร์ที่รองรับ:
    image_dir/
        Sushi/001.jpg, 002.jpg, ...
        Ramen/001.jpg, ...
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import time
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # โฟลเดอร์รูปที่มีโฟลเดอร์ย่อยตามเมนู
    "image_dir":  "C:\\Users\\ASUS\\Documents\\machine vision\\contest\\Intragram Images [Original]",

    # โฟลเดอร์ที่เก็บ model_sushi.pth ฯลฯ
    "model_dir":  "C:\\Users\\ASUS\\Documents\\machine vision\\contest\\checkpoints_per_menu",

    # จำนวนคู่ต่อรูป (round-robin)
    # 9000 รูป x N=5 → ~22,500 คู่/เมนู
    "pairs_per_image": 2,

    # กรอง confidence ต่ำออก
    "min_confidence": 0.55,

    # Batch size — RTX 2050 (4GB) ใช้ 32-64 ได้ ลดถ้า VRAM เต็ม
    "batch_size": 32,

    # นามสกุลไฟล์รูปที่รองรับ
    "image_extensions": [".jpg", ".jpeg", ".png", ".webp"],

    # Output
    "output_csv":     r"auto_labeled_data.csv",
    "checkpoint_dir": r"auto_label_checkpoints",

    # random seed
    "random_seed": 42,

    # image size (ต้องตรงกับที่ train ไว้)
    "image_size": 224,
}

MENU_FOLDER_MAP = {
    "Sushi":   "Sushi",
    "Ramen":   "Ramen",
    "Pizza":   "Pizza",
    "Burger":  "Burger",
    "Dessert": "Dessert",
}


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
class FoodSelectorNet(nn.Module):
    def __init__(self, model_name="efficientnet_b0", dropout=0.5):
        super().__init__()
        model_map = {
            "efficientnet_b0": (models.efficientnet_b0, 1280),
            "efficientnet_b1": (models.efficientnet_b1, 1280),
            "efficientnet_b2": (models.efficientnet_b2, 1408),
            "efficientnet_b3": (models.efficientnet_b3, 1536),
        }
        builder, feat_dim = model_map.get(model_name, (models.efficientnet_b0, 1280))
        base = builder(weights=None)
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


class NormalizeSixChannel:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406])
        self.std  = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225])

    def __call__(self, x):
        return (x - self.mean[:, None, None]) / self.std[:, None, None]


_normalizer = NormalizeSixChannel()


# ─────────────────────────────────────────────
#  PAIR DATASET สำหรับ batch prediction
# ─────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, pairs, image_size):
        self.pairs = pairs
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        try:
            img1 = Image.open(p1).convert("RGB")
            img2 = Image.open(p2).convert("RGB")
            combined = torch.cat([self.transform(img1), self.transform(img2)], dim=0)
            return combined, True
        except Exception:
            dummy = torch.zeros(6, self.image_size, self.image_size)
            return dummy, False


def collate_fn(batch):
    tensors, valids = zip(*batch)
    return _normalizer(torch.stack(tensors)), list(valids)


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
def load_all_models(model_dir, device):
    model_dir   = Path(model_dir)
    models_dict = {}
    pth_files   = list(model_dir.glob("model_*.pth"))

    if not pth_files:
        print(f"❌ ไม่พบ model ใน {model_dir}")
        return {}

    print(f"📦 โหลด Models จาก: {model_dir}")
    for pth in sorted(pth_files):
        ckpt = torch.load(pth, map_location=device)
        cfg  = ckpt.get("config", {})
        menu = ckpt.get("menu", pth.stem.replace("model_", "").capitalize())
        model = FoodSelectorNet(
            model_name=cfg.get("model_name", "efficientnet_b0"),
            dropout=cfg.get("dropout", 0.5),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models_dict[menu] = {"model": model, "cfg": cfg}
        print(f"   ✅ {menu:<10} | Val Acc: {ckpt.get('val_acc', 0)*100:.1f}%")

    return models_dict


# ─────────────────────────────────────────────
#  SCAN IMAGES
# ─────────────────────────────────────────────
def scan_menu_folders(image_dir):
    image_dir   = Path(image_dir)
    exts        = set(CONFIG["image_extensions"])
    menu_images = {}

    for folder in sorted(image_dir.iterdir()):
        if not folder.is_dir():
            continue
        menu   = MENU_FOLDER_MAP.get(folder.name, folder.name)
        images = [f for f in sorted(folder.iterdir())
                  if f.suffix.lower() in exts]
        if images:
            menu_images[menu] = images
            print(f"   📁 {folder.name:<12} → {len(images):,} รูป")
        else:
            print(f"   ⚠️  {folder.name} — ไม่พบรูปภาพ")

    return menu_images


# ─────────────────────────────────────────────
#  GENERATE PAIRS (round-robin)
# ─────────────────────────────────────────────
def generate_pairs(images, n_per_image, seed=42):
    random.seed(seed)
    imgs  = images.copy()
    random.shuffle(imgs)
    n     = len(imgs)
    pairs = set()
    for i in range(n):
        for offset in range(1, n_per_image + 1):
            j    = (i + offset) % n
            pair = (min(i, j), max(i, j))
            pairs.add(pair)
    return [(imgs[a], imgs[b]) for a, b in sorted(pairs)]


# ─────────────────────────────────────────────
#  CHECKPOINT
# ─────────────────────────────────────────────
def get_ckpt_path(menu):
    d = Path(CONFIG["checkpoint_dir"])
    d.mkdir(exist_ok=True)
    return d / f"ckpt_{menu.lower()}.json"


def load_checkpoint(menu):
    path = get_ckpt_path(menu)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"   ♻️  พบ checkpoint — รันต่อจาก batch {data['last_batch']+1} "
              f"(เก็บแล้ว {len(data['rows']):,} คู่)")
        return data["last_batch"], data["rows"]
    return -1, []


def save_checkpoint(menu, last_batch, rows):
    path = get_ckpt_path(menu)
    path.write_text(
        json.dumps({"last_batch": last_batch, "rows": rows}, ensure_ascii=False),
        encoding="utf-8"
    )


def delete_checkpoint(menu):
    path = get_ckpt_path(menu)
    if path.exists():
        path.unlink()


# ─────────────────────────────────────────────
#  BATCH PREDICT ONE MENU
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_menu(menu, images, model, cfg, device):
    image_size  = cfg.get("image_size", CONFIG["image_size"])
    pairs       = generate_pairs(images, CONFIG["pairs_per_image"], CONFIG["random_seed"])
    total_pairs = len(pairs)

    # โหลด checkpoint ถ้ามี
    last_batch, rows = load_checkpoint(menu)
    start_batch      = last_batch + 1
    kept             = len(rows)
    skipped          = 0

    dataset = PairDataset(pairs, image_size)
    loader  = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    total_batches = len(loader)
    t_start       = time.time()

    print(f"   จำนวนคู่  : {total_pairs:,}")
    print(f"   Batches   : {total_batches:,} (batch_size={CONFIG['batch_size']})")
    if start_batch > 0:
        print(f"   เริ่มจาก batch {start_batch}/{total_batches}")

    for batch_idx, (tensors, valids) in enumerate(loader):

        # ข้าม batch ที่เคยทำไปแล้ว
        if batch_idx < start_batch:
            continue

        tensors = tensors.to(device)
        probs   = torch.softmax(model(tensors), dim=1).cpu().numpy()
        base    = batch_idx * CONFIG["batch_size"]

        for i, (prob, valid) in enumerate(zip(probs, valids)):
            pair_idx = base + i
            if pair_idx >= total_pairs:
                break
            if not valid:
                skipped += 1
                continue

            winner     = int(np.argmax(prob)) + 1
            confidence = float(prob[winner - 1])

            if confidence < CONFIG["min_confidence"]:
                skipped += 1
                continue

            p1, p2 = pairs[pair_idx]
            kept  += 1
            num_vote_1 = round(float(prob[0]) * 100)
            num_vote_2 = round(float(prob[1]) * 100)
            rows.append({
                "Image 1":   p1.name,
                "Image 2":   p2.name,
                "Menu":      menu,
                "Winner":    winner,
                "Num Voter": 100,
                "Num Vote 1": num_vote_1,
                "Num Vote 2": num_vote_2,
            })

        # บันทึก checkpoint ทุก 50 batch
        if (batch_idx + 1) % 50 == 0:
            save_checkpoint(menu, batch_idx, rows)

        # แสดง progress ทุก 100 batch
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
            elapsed  = time.time() - t_start
            done     = batch_idx + 1 - start_batch
            total    = total_batches - start_batch
            eta      = (elapsed / max(done, 1)) * max(total - done, 0)
            print(f"   [{batch_idx+1:5d}/{total_batches}] "
                  f"เก็บ {kept:,} | ข้าม {skipped:,} | "
                  f"ผ่านไป {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    delete_checkpoint(menu)
    return rows, kept, skipped


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Auto Label — Food Image Pairs (Batch + Checkpoint)")
    print("=" * 60)
    print(f"  Image dir       : {CONFIG['image_dir']}")
    print(f"  Model dir       : {CONFIG['model_dir']}")
    print(f"  Pairs per image : {CONFIG['pairs_per_image']}")
    print(f"  Batch size      : {CONFIG['batch_size']}")
    print(f"  Min confidence  : {CONFIG['min_confidence']}")
    print(f"  Output CSV      : {CONFIG['output_csv']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📌 Device : {device}")
    if device.type == "cuda":
        print(f"   GPU  : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print()
    models_dict = load_all_models(CONFIG["model_dir"], device)
    if not models_dict:
        return

    print(f"\n📂 สแกนรูปจาก: {CONFIG['image_dir']}")
    menu_images = scan_menu_folders(CONFIG["image_dir"])
    if not menu_images:
        print("❌ ไม่พบรูปในโฟลเดอร์ใดเลย")
        return

    # ประมาณจำนวนคู่ทั้งหมด
    total_est = sum(
        len(generate_pairs(imgs, CONFIG["pairs_per_image"], CONFIG["random_seed"]))
        for imgs in menu_images.values()
    )
    print(f"\n   คาดว่าจะได้ประมาณ {total_est:,} คู่ทั้งหมด")

    # โหลด CSV เดิมถ้ามี (กรณีรันต่อ)
    output_path = Path(CONFIG["output_csv"])
    all_rows    = []
    done_menus  = set()

    if output_path.exists():
        existing   = pd.read_csv(output_path)
        done_menus = set(existing["Menu"].unique())
        all_rows   = existing.to_dict("records")
        print(f"\n⚠️  พบ CSV เดิม ({len(all_rows):,} คู่) — เมนูที่เสร็จแล้ว: {done_menus}")

    # ─── label แต่ละเมนู ───
    summary       = {}
    grand_kept    = 0
    grand_skipped = 0

    for menu, images in menu_images.items():
        if menu not in models_dict:
            print(f"\n⚠️  ไม่มี model สำหรับ '{menu}' — ข้ามไป")
            continue

        # ข้ามเมนูที่เสร็จแล้วและไม่มี checkpoint ค้าง
        ckpt_exists = get_ckpt_path(menu).exists()
        if menu in done_menus and not ckpt_exists:
            print(f"\n✅ {menu} — เสร็จแล้ว (ข้าม)")
            continue

        print(f"\n{'='*60}")
        print(f"  🍜 {menu} — {len(images):,} รูป")
        print(f"{'='*60}")

        t0 = time.time()
        rows, kept, skipped = predict_menu(
            menu, images,
            models_dict[menu]["model"],
            models_dict[menu]["cfg"],
            device
        )

        elapsed        = time.time() - t0
        grand_kept    += kept
        grand_skipped += skipped
        summary[menu]  = {"kept": kept, "skipped": skipped, "time": elapsed}

        # เพิ่ม rows ใหม่ (ไม่ซ้ำกับ done_menus เพราะข้ามไปแล้ว)
        all_rows.extend(rows)

        # บันทึก CSV ทันทีหลังเสร็จแต่ละเมนู
        cols = ["Image 1", "Image 2", "Menu", "Winner",
                "Num Voter", "Num Vote 1", "Num Vote 2"]
        pd.DataFrame(all_rows)[cols].to_csv(
            CONFIG["output_csv"], index=False, encoding="utf-8-sig"
        )
        print(f"\n  ✅ {menu} เสร็จ — เก็บ {kept:,} คู่ | ข้าม {skipped:,} | {elapsed/60:.1f} นาที")
        print(f"  💾 บันทึก CSV ชั่วคราว ({len(all_rows):,} คู่รวม)")

    # ─── สรุปผล ───
    print(f"\n{'='*60}")
    print(f"  📋 สรุปผล Auto Label")
    print(f"{'='*60}")
    print(f"  {'เมนู':<10} {'เก็บ':>8} {'ข้าม':>8} {'เวลา':>8}")
    print(f"  {'-'*38}")
    for menu, s in summary.items():
        print(f"  {menu:<10} {s['kept']:>8,} {s['skipped']:>8,} {s['time']/60:>7.1f}m")
    print(f"  {'-'*38}")
    print(f"  {'รวม':<10} {grand_kept:>8,} {grand_skipped:>8,}")

    result_df = pd.DataFrame(all_rows)
    if len(result_df):
        print(f"\n  Winner distribution:")
        for menu, grp in result_df.groupby("Menu"):
            w1       = (grp["Winner"] == 1).sum()
            w2       = (grp["Winner"] == 2).sum()
            # คำนวณ avg confidence จาก Num Vote ของฝั่งที่ชนะ
            avg_conf = (grp[["Num Vote 1", "Num Vote 2"]].max(axis=1) / 100).mean()
            print(f"     {menu:<10} : W1={w1:,} | W2={w2:,} | avg conf={avg_conf:.3f}")

    print(f"\n  💾 Output CSV : {CONFIG['output_csv']}")
    print(f"  📌 CSV นี้ใช้เป็น dataset สำหรับ train ต่อได้เลย")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()