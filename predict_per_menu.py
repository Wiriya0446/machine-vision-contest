"""
Food Image Pair Selector - Inference (Per-Menu Models)
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

#python predict_per_menu.py --csv "C:\Users\ASUS\Documents\machine vision\contest\Test Set Samples\Test_IG.csv" --img_dir "Questionaire Images" --img_dir2 "C:\Users\ASUS\Documents\machine vision\contest\Intragram Images [Original]"
# ─────────────────────────────────────────────
#  CONFIG 
# ─────────────────────────────────────────────
DEFAULT_MODEL_DIR  = r"checkpoints_per_menu"
DEFAULT_CSV_PATH   = r"data_from_questionaire.csv"
DEFAULT_IMAGE_DIR  = r"Questionaire Images"          # แหล่งรูปที่ 1
DEFAULT_IMAGE_DIR2 = r""                              # แหล่งรูปที่ 2 (ใส่ "" เพื่อปิด)
DEFAULT_OUTPUT_CSV = r"prediction_results_per_menu.csv"


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
class FoodSelectorNet(nn.Module):
    def __init__(self, model_name="efficientnet_b0", dropout=0.5, pretrained=False):
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
#  FIND IMAGE — ค้นหาจาก 2 โฟลเดอร์
# ─────────────────────────────────────────────
def find_image(filename, img_dir1, img_dir2=None):
    """
    ค้นหารูปจาก img_dir1 ก่อน ถ้าไม่เจอค่อยหาใน img_dir2
    คืน Path ที่เจอ หรือ None ถ้าไม่พบทั้งคู่
    """
    for d in [img_dir1, img_dir2]:
        if d is None:
            continue
        p = Path(d) / filename
        if p.exists():
            return p
        # ค้นหาในโฟลเดอร์ย่อย
        matches = list(Path(d).rglob(filename))
        if matches:
            return matches[0]
    return None


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
def load_all_models(model_dir, device):
    model_dir = Path(model_dir)
    models_dict = {}
    pth_files = list(model_dir.glob("model_*.pth"))
    if not pth_files:
        print(f"❌ ไม่พบไฟล์ model ใน {model_dir}")
        sys.exit(1)

    print(f"📦 โหลด Per-Menu Models จาก: {model_dir}")
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
#  PREDICT
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_pair(model, device, cfg, img1_path, img2_path):
    transform = transforms.Compose([
        transforms.Resize((cfg.get("image_size", 224), cfg.get("image_size", 224))),
        transforms.ToTensor(),
    ])
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    combined = _normalizer(torch.cat([transform(img1), transform(img2)], dim=0).unsqueeze(0)).to(device)
    probs  = torch.softmax(model(combined), dim=1)[0].cpu().numpy()
    winner = int(np.argmax(probs)) + 1
    return {"predicted_winner": winner,
            "confidence": float(probs[winner - 1]),
            "prob_img1":  float(probs[0]),
            "prob_img2":  float(probs[1])}


# ─────────────────────────────────────────────
#  PREDICT FROM CSV
# ─────────────────────────────────────────────
def predict_from_csv(models_dict, device, csv_path, image_dir, image_dir2, output_csv):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    has_label = "Winner" in df.columns
    img_dir2  = image_dir2 if image_dir2 else None

    print(f"\n{'='*60}")
    print(f"  CSV         : {csv_path}")
    print(f"  Image dir 1 : {image_dir}")
    print(f"  Image dir 2 : {img_dir2 or '(ไม่ได้ใช้)'}")
    print(f"  จำนวนข้อมูล  : {len(df)} คู่")
    print(f"{'='*60}")

    results = []
    correct, total, skipped, no_model = 0, 0, 0, 0

    for i, row in df.iterrows():
        menu = row.get("Menu", "")

        # ค้นหารูปจากทั้ง 2 โฟลเดอร์
        p1 = find_image(row["Image 1"], image_dir, img_dir2)
        p2 = find_image(row["Image 2"], image_dir, img_dir2)

        if p1 is None or p2 is None:
            missing = row["Image 1"] if p1 is None else row["Image 2"]
            print(f"  [{i+1:4d}] WARNING ไม่พบรูป: {missing}")
            skipped += 1
            results.append({**row.to_dict(), "predicted_winner": None,
                             "confidence": None, "prob_img1": None,
                             "prob_img2": None, "correct": None, "model_used": None})
            continue

        if menu not in models_dict:
            print(f"  [{i+1:4d}] WARNING ไม่มี model สำหรับเมนู '{menu}'")
            no_model += 1
            results.append({**row.to_dict(), "predicted_winner": None,
                             "confidence": None, "prob_img1": None,
                             "prob_img2": None, "correct": None, "model_used": None})
            continue

        entry      = models_dict[menu]
        res        = predict_pair(entry["model"], device, entry["cfg"], str(p1), str(p2))
        predicted  = res["predicted_winner"]
        actual     = int(row["Winner"]) if has_label else None
        is_correct = (predicted == actual) if has_label else None

        if has_label:
            correct += int(is_correct)
        total += 1

        status = ("✓" if is_correct else "✗") if has_label else ""
        print(f"  [{i+1:4d}] {menu:8s} | {row['Image 1']:15s} vs {row['Image 2']:15s} "
              f"-> รูปที่ {predicted} ({res['confidence']*100:.1f}%) {status}")

        results.append({**row.to_dict(),
                        "predicted_winner": predicted,
                        "confidence":       round(res["confidence"], 4),
                        "prob_img1":        round(res["prob_img1"], 4),
                        "prob_img2":        round(res["prob_img2"], 4),
                        "correct":          is_correct,
                        "model_used":       f"model_{menu.lower()}.pth"})

    # สรุป
    print(f"\n{'='*60}")
    print(f"  ทำนายทั้งหมด  : {total} คู่")
    print(f"  ไม่พบรูป      : {skipped} คู่")
    print(f"  ไม่มี model   : {no_model} คู่")
    if has_label and total > 0:
        print(f"  ถูกต้อง       : {correct}/{total} คู่")
        print(f"  Accuracy      : {correct/total*100:.2f}%")
        print(f"\n  Accuracy แยกตามเมนู:")
        rdf = pd.DataFrame(results).dropna(subset=["predicted_winner"])
        for menu, grp in rdf.groupby("Menu"):
            acc = grp["correct"].mean()
            print(f"     {menu:10s} : {acc*100:.1f}%  ({int(grp['correct'].sum())}/{len(grp)})")

    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n  บันทึกผลลัพธ์ที่ : {output_csv}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Food Selector Predict (Per-Menu)")
    parser.add_argument("--model_dir",  default=DEFAULT_MODEL_DIR)
    parser.add_argument("--csv",        default=DEFAULT_CSV_PATH)
    parser.add_argument("--img_dir",    default=DEFAULT_IMAGE_DIR,   help="แหล่งรูปที่ 1")
    parser.add_argument("--img_dir2",   default=DEFAULT_IMAGE_DIR2,  help="แหล่งรูปที่ 2 (ถ้ามี)")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    models_dict = load_all_models(args.model_dir, device)
    predict_from_csv(models_dict, device, args.csv,
                     args.img_dir, args.img_dir2, args.output)


if __name__ == "__main__":
    main()