"""
Food Image Pair Selector — Inference (Per-Menu Models)
=======================================================
ใช้ model แบบ 5 ไฟล์แยกเมนู ทำนายว่ารูปไหนชนะ
Input CSV  : Image 1, Image 2, Menu  (ไม่ต้องมีคอลัมน์ Winner)
Output CSV : Image 1, Image 2, Menu, Winner  (format เดียวกับ training data)

python predict_per_menu.py --csv "test.csv" --img_dir "Images"
python predict_per_menu.py --csv "test.csv" --img_dir "Images" --img_dir2 "Images2" --img_dir3 "Images3"
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  DEFAULT CONFIG
# ─────────────────────────────────────────────
DEFAULT_MODEL_DIR  = r"checkpoints_per_menu"
DEFAULT_CSV_PATH   = r"test.csv"
DEFAULT_IMAGE_DIR  = r"Questionaire Images"
DEFAULT_IMAGE_DIR2 = r""
DEFAULT_IMAGE_DIR3 = r""
DEFAULT_OUTPUT_CSV = r"prediction_results_per_menu.csv"


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


# ─────────────────────────────────────────────
#  NORMALIZE
# ─────────────────────────────────────────────
class NormalizeSixChannel:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406])
        self.std  = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225])

    def __call__(self, x):
        return (x - self.mean[:, None, None]) / self.std[:, None, None]


_normalizer = NormalizeSixChannel()


# ─────────────────────────────────────────────
#  FIND IMAGE — ค้นหาจากหลายโฟลเดอร์
# ─────────────────────────────────────────────
def find_image(filename, *dirs):
    for d in dirs:
        if not d:
            continue
        p = Path(d) / filename
        if p.exists():
            return p
        matches = list(Path(d).rglob(filename))
        if matches:
            return matches[0]
    return None


# ─────────────────────────────────────────────
#  LOAD ALL MODELS
# ─────────────────────────────────────────────
def load_all_models(model_dir, device):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        print(f"❌ ไม่พบโฟลเดอร์ model: {model_dir}")
        sys.exit(1)

    pth_files = list(model_dir.glob("model_*.pth"))
    if not pth_files:
        print(f"❌ ไม่พบไฟล์ model (model_*.pth) ใน {model_dir}")
        sys.exit(1)

    print(f"📦 โหลด Per-Menu Models จาก: {model_dir}")
    models_dict = {}
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
#  PREDICT ONE PAIR
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_pair(model, device, cfg, img1_path, img2_path):
    size = cfg.get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    combined = _normalizer(
        torch.cat([transform(img1), transform(img2)], dim=0)
    ).unsqueeze(0).to(device)

    probs  = torch.softmax(model(combined), dim=1)[0].cpu().numpy()
    winner = int(np.argmax(probs)) + 1  # 1 หรือ 2

    return {
        "winner":     winner,
        "confidence": float(probs[winner - 1]),
        "prob_img1":  float(probs[0]),
        "prob_img2":  float(probs[1]),
    }


# ─────────────────────────────────────────────
#  PREDICT FROM CSV
# ─────────────────────────────────────────────
def predict_from_csv(models_dict, device, csv_path,
                     img_dir, img_dir2, img_dir3, output_csv):

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # ตรวจสอบคอลัมน์ที่จำเป็น
    for col in ["Image 1", "Image 2", "Menu"]:
        if col not in df.columns:
            print(f"❌ ไม่พบคอลัมน์ '{col}' ใน CSV")
            sys.exit(1)

    # ถ้ามีคอลัมน์ Winner อยู่แล้ว ให้เก็บไว้เปรียบเทียบได้
    has_label = "Winner" in df.columns

    print(f"\n{'='*60}")
    print(f"  CSV         : {csv_path}")
    print(f"  Image dir 1 : {img_dir}")
    print(f"  Image dir 2 : {img_dir2 or '(ไม่ได้ใช้)'}")
    print(f"  Image dir 3 : {img_dir3 or '(ไม่ได้ใช้)'}")
    print(f"  จำนวนข้อมูล  : {len(df)} คู่")
    print(f"  มีเฉลย       : {'ใช่' if has_label else 'ไม่มี'}")
    print(f"{'='*60}")

    winners = []
    skipped = 0
    correct = 0
    total   = 0

    for i, row in df.iterrows():
        menu = str(row.get("Menu", "")).strip()

        p1 = find_image(str(row["Image 1"]).strip(), img_dir, img_dir2, img_dir3)
        p2 = find_image(str(row["Image 2"]).strip(), img_dir, img_dir2, img_dir3)

        # ไม่พบรูป
        if p1 is None or p2 is None:
            missing = row["Image 1"] if p1 is None else row["Image 2"]
            print(f"  [{i+1:4d}] ⚠ ไม่พบรูป: {missing}")
            winners.append(None)
            skipped += 1
            continue

        # ไม่มี model สำหรับเมนูนี้
        if menu not in models_dict:
            print(f"  [{i+1:4d}] ⚠ ไม่มี model สำหรับเมนู '{menu}'")
            winners.append(None)
            skipped += 1
            continue

        entry = models_dict[menu]
        res   = predict_pair(entry["model"], device, entry["cfg"], str(p1), str(p2))

        winners.append(res["winner"])
        total += 1

        # แสดงผล
        if has_label:
            actual     = int(row["Winner"])
            is_correct = res["winner"] == actual
            correct   += int(is_correct)
            mark       = "✓" if is_correct else "✗"
            print(f"  [{i+1:4d}] {menu:8s} | รูปที่ {res['winner']} ({res['confidence']*100:.1f}%) "
                  f"{mark}  (เฉลย: {actual})")
        else:
            print(f"  [{i+1:4d}] {menu:8s} | รูปที่ {res['winner']} ({res['confidence']*100:.1f}%)")

    # ─── สร้าง Output CSV (format เดียวกับ training data) ───────────
    out_df = df[["Image 1", "Image 2", "Menu"]].copy()
    out_df["Winner"] = winners
    out_df["Winner"] = out_df["Winner"].fillna(0).astype(int)

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # ─── สรุปผล ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ทำนายสำเร็จ   : {total} คู่")
    print(f"  ข้ามไป        : {skipped} คู่  (ไม่พบรูป/ไม่มี model)")

    if has_label and total > 0:
        print(f"  ถูกต้อง       : {correct}/{total} คู่")
        print(f"  Accuracy      : {correct/total*100:.2f}%")
        print(f"\n  Accuracy แยกตามเมนู:")
        result_df = out_df.copy()
        result_df["actual"] = df["Winner"].values
        result_df = result_df[result_df["Winner"] != 0]
        for menu, grp in result_df.groupby("Menu"):
            acc = (grp["Winner"] == grp["actual"]).mean()
            n   = (grp["Winner"] == grp["actual"]).sum()
            print(f"     {menu:10s} : {acc*100:.1f}%  ({n}/{len(grp)})")

    print(f"\n  บันทึกผลที่    : {output_csv}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Food Selector Inference (Per-Menu Models)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                        help=f"โฟลเดอร์ที่เก็บ model_*.pth  (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--csv",       default=DEFAULT_CSV_PATH,
                        help=f"input CSV (Image 1, Image 2, Menu)  (default: {DEFAULT_CSV_PATH})")
    parser.add_argument("--img_dir",   default=DEFAULT_IMAGE_DIR,
                        help="โฟลเดอร์รูปที่ 1")
    parser.add_argument("--img_dir2",  default=DEFAULT_IMAGE_DIR2,
                        help="โฟลเดอร์รูปที่ 2 (ถ้ามี)")
    parser.add_argument("--img_dir3",  default=DEFAULT_IMAGE_DIR3,
                        help="โฟลเดอร์รูปที่ 3 (ถ้ามี)")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT_CSV,
                        help=f"output CSV  (default: {DEFAULT_OUTPUT_CSV})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    models_dict = load_all_models(args.model_dir, device)

    predict_from_csv(
        models_dict, device,
        csv_path  = args.csv,
        img_dir   = args.img_dir,
        img_dir2  = args.img_dir2 or None,
        img_dir3  = args.img_dir3 or None,
        output_csv= args.output,
    )


if __name__ == "__main__":
    main()