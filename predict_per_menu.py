"""
Food Image Pair Selector — Two-Stage Inference
===============================================
Pipeline:
  Stage 1 : Food Classifier (Triplet Loss)  → แยกชนิดอาหารจากรูป
  Stage 2 : Per-Menu Model (EfficientNet)   → เลือกรูปที่น่ากินกว่า

ใช้เมื่อ CSV ไม่มีคอลัมน์ Menu มาให้

วิธีใช้:
  python predict_per_menu.py \
    --classifier  "checkpoints_classifier/food_classifier.pth" \
    --gallery_dir "Intragram Images [Original]" \
    --model_dir   "checkpoints_per_menu" \
    --csv         "Test_IG.csv" \
    --img_dir     "Questionaire Images" \
    --img_dir2    "Intragram Images [Original]" \
    --output      "prediction_results_per_menu.csv"
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  DEFAULT CONFIG
# ─────────────────────────────────────────────
DEFAULT_CLASSIFIER  = r"checkpoints_classifier\food_classifier.pth"
DEFAULT_GALLERY_DIR = r"C:\Users\ASUS\Documents\machine vision\contest\Intragram Images [Original]"
DEFAULT_GALLERY_N   = 100     # รูปต่อเมนูที่ใช้สร้าง gallery (None = ทั้งหมด)
DEFAULT_KNN         = 5
DEFAULT_MODEL_DIR   = r"checkpoints_per_menu"
DEFAULT_CSV_PATH    = r"data_from_questionaire.csv"
DEFAULT_IMAGE_DIR   = r"Questionaire Images"
DEFAULT_IMAGE_DIR2  = r""
DEFAULT_OUTPUT_CSV  = r"prediction_results_per_menu.csv"


# ─────────────────────────────────────────────
#  IMAGE INDEX — scan ครั้งเดียว lookup O(1)
# ─────────────────────────────────────────────
def build_image_index(dirs):
    exts  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    index = {}
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"  [WARNING] ไม่พบโฟลเดอร์: {d}")
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in exts and f.name not in index:
                index[f.name] = f
    return index


def load_pil(filename, image_index):
    p = image_index.get(Path(filename).name)
    if p is None:
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


# ═════════════════════════════════════════════
#  STAGE 1 — FOOD CLASSIFIER (Triplet kNN)
# ═════════════════════════════════════════════
class FoodEmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.backbone = base.features
        self.pool     = base.avgpool
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return F.normalize(self.head(x), p=2, dim=1)


def get_clf_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


def load_classifier(model_path, device):
    ckpt       = torch.load(model_path, map_location=device)
    cfg        = ckpt.get("config", {})
    menus      = ckpt["menus"]
    embed_dim  = cfg.get("embed_dim",  128)
    backbone   = cfg.get("backbone",   "efficientnet_b0")
    dropout    = cfg.get("dropout",    0.3)
    image_size = cfg.get("image_size", 224)

    model = FoodEmbeddingNet(embed_dim=embed_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  [Classifier] menus      : {menus}")
    print(f"  [Classifier] val_acc    : {ckpt.get('val_acc', 0)*100:.1f}%")
    return model, menus, image_size


@torch.no_grad()
def build_gallery(clf_model, gallery_dir, menus, transform, device, gallery_n=None):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    rng  = random.Random(42)
    all_emb, all_lbl = [], []
    menu_to_idx = {m: i for i, m in enumerate(menus)}

    print(f"\n  Building gallery จาก: {gallery_dir}")
    for folder in sorted(Path(gallery_dir).iterdir()):
        if not folder.is_dir() or folder.name not in menus:
            continue
        imgs = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in exts]
        if gallery_n and len(imgs) > gallery_n:
            imgs = rng.sample(imgs, gallery_n)

        tensors = []
        for p in imgs:
            try:
                tensors.append(transform(Image.open(p).convert("RGB")))
            except Exception:
                continue
        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        emb   = clf_model(batch).cpu()
        all_emb.append(emb)
        all_lbl.extend([menu_to_idx[folder.name]] * len(tensors))
        print(f"    {folder.name:<12} {len(tensors):>5,} รูป")

    gallery_emb = torch.cat(all_emb)
    gallery_lbl = torch.tensor(all_lbl)
    print(f"  Gallery รวม: {len(gallery_lbl):,} embeddings")
    return gallery_emb, gallery_lbl


@torch.no_grad()
def classify_menu(img_pil, clf_model, gallery_emb, gallery_lbl,
                  menus, transform, device, k=5):
    """คืน (predicted_menu, confidence)"""
    t   = transform(img_pil).unsqueeze(0).to(device)
    emb = clf_model(t).cpu()

    sim   = torch.mm(emb, gallery_emb.t()).squeeze(0)
    topk  = sim.topk(k)
    top_labels = gallery_lbl[topk.indices]
    top_sims   = topk.values

    scores = defaultdict(float)
    for lbl, s in zip(top_labels.tolist(), top_sims.tolist()):
        scores[lbl] += s

    winner_idx  = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence  = scores[winner_idx] / total_score if total_score > 0 else 0.0
    return menus[winner_idx], round(confidence, 4)


# ═════════════════════════════════════════════
#  STAGE 2 — PER-MENU SELECTOR (EfficientNet)
# ═════════════════════════════════════════════
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
        old  = base.features[0][0]
        new  = nn.Conv2d(6, old.out_channels, old.kernel_size,
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
        return self.classifier(torch.flatten(x, 1))


class NormalizeSixChannel:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406])
        self.std  = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225])

    def __call__(self, x):
        return (x - self.mean[:, None, None]) / self.std[:, None, None]


_normalizer = NormalizeSixChannel()


def get_sel_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def load_selector_models(model_dir, device):
    """โหลด model ทุกเมนูจาก checkpoints_per_menu/"""
    model_dir   = Path(model_dir)
    models_dict = {}
    for pth in sorted(model_dir.glob("model_*.pth")):
        ckpt  = torch.load(pth, map_location=device)
        cfg   = ckpt.get("config", {})
        menu  = ckpt.get("menu", pth.stem.replace("model_", "").capitalize())
        model = FoodSelectorNet(
            cfg.get("model_name", "efficientnet_b0"),
            cfg.get("dropout", 0.5)
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models_dict[menu] = {
            "model":      model,
            "image_size": cfg.get("image_size", 224),
            "val_acc":    ckpt.get("val_acc", 0),
        }
        print(f"  [Selector] {menu:<10} val_acc={ckpt.get('val_acc',0)*100:.1f}%")
    return models_dict


@torch.no_grad()
def select_winner(img1_pil, img2_pil, selector_model, image_size, device):
    """คืน (winner=1or2, confidence, prob1, prob2)"""
    tf = get_sel_transform(image_size)
    t1 = tf(img1_pil)
    t2 = tf(img2_pil)
    combined = _normalizer(torch.cat([t1, t2], dim=0)).unsqueeze(0).to(device)
    logits   = selector_model(combined)
    probs    = torch.softmax(logits, dim=1).squeeze(0).cpu()
    winner   = int(probs.argmax().item()) + 1
    return winner, round(float(probs[winner-1]), 4), round(float(probs[0]), 4), round(float(probs[1]), 4)


# ═════════════════════════════════════════════
#  TWO-STAGE PIPELINE
# ═════════════════════════════════════════════
def predict_from_csv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ── โหลด Stage 1: Classifier ──
    print(f"\n  {'─'*55}")
    print(f"  Stage 1 — Food Classifier")
    print(f"  {'─'*55}")
    clf_model, menus, clf_img_size = load_classifier(args.classifier, device)
    clf_transform = get_clf_transform(clf_img_size)
    gallery_emb, gallery_lbl = build_gallery(
        clf_model, args.gallery_dir, menus,
        clf_transform, device, gallery_n=args.gallery_n
    )

    # ── โหลด Stage 2: Per-Menu Selector ──
    print(f"\n  {'─'*55}")
    print(f"  Stage 2 — Per-Menu Selector")
    print(f"  {'─'*55}")
    selector_models = load_selector_models(args.model_dir, device)
    if not selector_models:
        print(f"  ❌ ไม่พบ model ใน {args.model_dir}")
        return

    # ── Image Index ──
    img_dirs    = [d for d in [args.img_dir, args.img_dir2] if d]
    image_index = build_image_index(img_dirs)
    print(f"\n  Image index: {len(image_index):,} รูป")

    # ── โหลด CSV ──
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    has_label  = "Winner" in df.columns
    has_menu   = "Menu"   in df.columns

    print(f"\n{'='*60}")
    print(f"  CSV         : {args.csv}")
    print(f"  จำนวนข้อมูล  : {len(df)} คู่")
    print(f"  มีคอลัมน์ Menu  : {'ใช่' if has_menu else 'ไม่มี — ใช้ Classifier แยกเมนู'}")
    print(f"  มีคอลัมน์ Winner: {'ใช่ (วัด accuracy ได้)' if has_label else 'ไม่มี'}")
    print(f"{'='*60}")

    results = []
    correct_winner = 0
    correct_menu1  = 0
    correct_menu2  = 0
    total = skipped = fallback = 0

    for i, row in df.iterrows():
        img1_name = row["Image 1"]
        img2_name = row["Image 2"]
        true_menu = row.get("Menu", "") if has_menu else ""

        # โหลดรูป
        img1_pil = load_pil(img1_name, image_index)
        img2_pil = load_pil(img2_name, image_index)

        if img1_pil is None or img2_pil is None:
            missing = img1_name if img1_pil is None else img2_name
            print(f"  [{i+1:4d}] WARNING ไม่พบรูป: {missing}")
            skipped += 1
            results.append({
                "Image 1": row["Image 1"],
                "Image 2": row["Image 2"],
                "Winner":  None,
            })
            continue

        # ── Stage 1: แยกเมนู ──
        if has_menu:
            # CSV มีเมนูอยู่แล้ว ข้าม classifier
            pred_menu   = true_menu
            menu_conf1  = 1.0
            menu_conf2  = 1.0
        else:
            # ให้ classifier แยกเมนูของแต่ละรูป แล้วโหวต
            menu1, conf1 = classify_menu(img1_pil, clf_model, gallery_emb,
                                         gallery_lbl, menus, clf_transform, device, args.k)
            menu2, conf2 = classify_menu(img2_pil, clf_model, gallery_emb,
                                         gallery_lbl, menus, clf_transform, device, args.k)
            menu_conf1, menu_conf2 = conf1, conf2

            # โหวตจากทั้ง 2 รูป — ถ้าตรงกันใช้เมนูนั้น ถ้าไม่ตรงใช้อันที่ confidence สูงกว่า
            pred_menu = menu1 if (menu1 == menu2 or conf1 >= conf2) else menu2

        # ── Stage 2: เลือกรูปที่ดีกว่า ──
        sel_info = selector_models.get(pred_menu)
        if sel_info is None:
            # ไม่มี model สำหรับเมนูนี้ — fallback: เลือกรูปที่ 1
            winner, conf_w, prob1, prob2 = 1, 0.5, 0.5, 0.5
            model_used = f"fallback(no_model_{pred_menu})"
            fallback  += 1
        else:
            winner, conf_w, prob1, prob2 = select_winner(
                img1_pil, img2_pil,
                sel_info["model"], sel_info["image_size"], device
            )
            model_used = f"selector_{pred_menu.lower()}"

        # ── ตรวจความถูกต้อง ──
        true_winner = int(row["Winner"]) if has_label else None
        is_correct  = (winner == true_winner) if has_label else None
        if has_label:
            correct_winner += int(is_correct)
        if has_menu and not has_menu:   # เฉพาะกรณีที่ classify เอง
            pass
        if not has_menu:
            if menu1 == true_menu: correct_menu1 += 1
            if menu2 == true_menu: correct_menu2 += 1
        total += 1

        # log
        menu_tag  = f"[{pred_menu}]" if not has_menu else f"[{pred_menu}✓]" if pred_menu == true_menu else f"[{pred_menu}✗→{true_menu}]"
        win_tag   = ("✓" if is_correct else "✗") if has_label else ""
        print(f"  [{i+1:4d}] {menu_tag:14s} | "
              f"{img1_name:15s} vs {img2_name:15s} "
              f"→ รูปที่ {winner} {win_tag}")

        results.append({
            "Image 1": row["Image 1"],
            "Image 2": row["Image 2"],
            "Winner":  winner,
        })

    # ── บันทึก CSV ──
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    # ── สรุปผล ──
    print(f"\n{'='*60}")
    print(f"  ทำนายทั้งหมด  : {total} คู่")
    print(f"  ไม่พบรูป      : {skipped} คู่")
    if fallback > 0:
        print(f"  Fallback      : {fallback} คู่ (ไม่มี selector model)")
    if has_label and total > 0:
        print(f"  Winner Acc    : {correct_winner}/{total} ({correct_winner/total*100:.1f}%)")
        print(f"\n  Winner Accuracy แยกตามเมนู:")
        rdf = out_df.dropna(subset=["predicted_winner"])
        for menu, grp in rdf.groupby("predicted_menu"):
            acc = grp["correct"].mean()
            print(f"     {menu:<10} : {acc*100:.1f}%  ({int(grp['correct'].sum())}/{len(grp)})")
    print(f"\n  บันทึกผลลัพธ์ที่: {args.output}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Food Pair Selector — Two-Stage (Classifier + Per-Menu Model)"
    )
    # Stage 1
    parser.add_argument("--classifier",  default=DEFAULT_CLASSIFIER,
                        help="path ของ food_classifier.pth")
    parser.add_argument("--gallery_dir", default=DEFAULT_GALLERY_DIR,
                        help="โฟลเดอร์รูปสำหรับสร้าง gallery (มีโฟลเดอร์ย่อยตามเมนู)")
    parser.add_argument("--gallery_n",   type=int, default=DEFAULT_GALLERY_N,
                        help="รูปต่อเมนูที่ใช้สร้าง gallery")
    parser.add_argument("--k",           type=int, default=DEFAULT_KNN,
                        help="kNN neighbors สำหรับแยกเมนู")
    # Stage 2
    parser.add_argument("--model_dir",   default=DEFAULT_MODEL_DIR,
                        help="โฟลเดอร์ที่เก็บ model_sushi.pth ฯลฯ")
    # Input/Output
    parser.add_argument("--csv",         default=DEFAULT_CSV_PATH)
    parser.add_argument("--img_dir",     default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--img_dir2",    default=DEFAULT_IMAGE_DIR2)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    if args.img_dir2 == "":
        args.img_dir2 = None

    predict_from_csv(args)


if __name__ == "__main__":
    main()