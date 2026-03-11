"""
Filter & Balance Dataset
========================
กรอง auto_labeled_data.csv ด้วย confidence threshold
และ balance จำนวนคู่แต่ละเมนูให้เท่ากัน

วิธีใช้:
    python filter_and_balance.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # Input — CSV จาก auto_label.py
    "input_csv":  "C:\\Users\\ASUS\\Documents\\machine vision\\contest\\auto_labeled_data.csv",

    # Output
    "output_csv": r"auto_labeled_filtered.csv",

    # กรอง confidence ต่ำออก
    # Num Vote ของฝั่งชนะต้องมากกว่านี้ (จาก 100)
    # เช่น 65 = model มั่นใจอย่างน้อย 65% ถึงเก็บ
    "min_confidence_vote": 65,

    # Balance — จำนวนคู่สูงสุดต่อเมนู
    # None = ใช้จำนวนของเมนูที่มีน้อยที่สุดเป็นเกณฑ์
    # ใส่ตัวเลข เช่น 500 = จำกัดทุกเมนูไม่เกิน 500 คู่
    "max_pairs_per_menu": None,

    # สุ่มแบบ reproducible
    "random_seed": 42,
}


def main():
    print("=" * 60)
    print("  Filter & Balance Dataset")
    print("=" * 60)
    print(f"  Input          : {CONFIG['input_csv']}")
    print(f"  Min confidence : {CONFIG['min_confidence_vote']}/100")
    print(f"  Max pairs/menu : {CONFIG['max_pairs_per_menu'] or 'auto (min เมนู)'}")

    # ── โหลด CSV ──
    df = pd.read_csv(CONFIG["input_csv"])
    df.columns = df.columns.str.strip()
    print(f"\n  โหลดข้อมูล: {len(df):,} คู่")

    # ── แสดงสถานะก่อน filter ──
    print(f"\n  ก่อน filter:")
    print(f"  {'เมนู':<12} {'จำนวน':>8} {'avg conf':>10} {'W1':>6} {'W2':>6}")
    print(f"  {'-'*45}")
    for menu, grp in df.groupby("Menu"):
        avg_conf = grp[["Num Vote 1", "Num Vote 2"]].max(axis=1).mean()
        w1 = (grp["Winner"] == 1).sum()
        w2 = (grp["Winner"] == 2).sum()
        print(f"  {menu:<12} {len(grp):>8,} {avg_conf:>9.1f} {w1:>6,} {w2:>6,}")

    # ── Step 1: Filter confidence ──
    # คำนวณ confidence จาก Num Vote ของฝั่งที่ชนะ
    df["_win_vote"] = df.apply(
        lambda r: r["Num Vote 1"] if r["Winner"] == 1 else r["Num Vote 2"], axis=1
    )
    before = len(df)
    df = df[df["_win_vote"] >= CONFIG["min_confidence_vote"]].copy()
    df.drop(columns=["_win_vote"], inplace=True)
    after_filter = len(df)

    print(f"\n  หลัง filter (confidence >= {CONFIG['min_confidence_vote']}):")
    print(f"  เหลือ {after_filter:,} คู่ (ตัดออก {before - after_filter:,} คู่)")
    print(f"\n  {'เมนู':<12} {'จำนวน':>8} {'avg conf':>10} {'W1':>6} {'W2':>6}")
    print(f"  {'-'*45}")
    for menu, grp in df.groupby("Menu"):
        avg_conf = grp[["Num Vote 1", "Num Vote 2"]].max(axis=1).mean()
        w1 = (grp["Winner"] == 1).sum()
        w2 = (grp["Winner"] == 2).sum()
        print(f"  {menu:<12} {len(grp):>8,} {avg_conf:>9.1f} {w1:>6,} {w2:>6,}")

    if len(df) == 0:
        print("\n  ❌ ไม่มีข้อมูลเหลือหลัง filter — ลด min_confidence_vote ลงครับ")
        return

    # ── Step 2: Balance ──
    menu_counts = df.groupby("Menu").size()
    min_count   = menu_counts.min()
    cap         = CONFIG["max_pairs_per_menu"] or min_count

    print(f"\n  Balance — จำกัดแต่ละเมนูไม่เกิน {cap:,} คู่")

    balanced_parts = []
    for menu, grp in df.groupby("Menu"):
        if len(grp) > cap:
            # สุ่มเลือกให้ได้ balance ระหว่าง Winner=1 และ Winner=2 ด้วย
            half = cap // 2
            w1   = grp[grp["Winner"] == 1].sample(
                n=min(half, (grp["Winner"] == 1).sum()),
                random_state=CONFIG["random_seed"]
            )
            w2   = grp[grp["Winner"] == 2].sample(
                n=min(cap - len(w1), (grp["Winner"] == 2).sum()),
                random_state=CONFIG["random_seed"]
            )
            # ถ้า w1 หรือ w2 น้อยกว่า half ให้เติมจากอีกฝั่ง
            if len(w1) + len(w2) < cap:
                remaining = cap - len(w1) - len(w2)
                extra_pool = grp.drop(index=pd.concat([w1, w2]).index)
                if len(extra_pool) > 0:
                    extra = extra_pool.sample(
                        n=min(remaining, len(extra_pool)),
                        random_state=CONFIG["random_seed"]
                    )
                    grp_sampled = pd.concat([w1, w2, extra])
                else:
                    grp_sampled = pd.concat([w1, w2])
            else:
                grp_sampled = pd.concat([w1, w2])
        else:
            grp_sampled = grp

        balanced_parts.append(grp_sampled)

    result_df = pd.concat(balanced_parts, ignore_index=True)
    result_df  = result_df.sample(frac=1, random_state=CONFIG["random_seed"]).reset_index(drop=True)

    # ── สรุป ──
    print(f"\n  หลัง balance:")
    print(f"  {'เมนู':<12} {'จำนวน':>8} {'W1':>6} {'W2':>6} {'W1%':>6}")
    print(f"  {'-'*40}")
    for menu, grp in result_df.groupby("Menu"):
        w1  = (grp["Winner"] == 1).sum()
        w2  = (grp["Winner"] == 2).sum()
        pct = w1 / len(grp) * 100
        print(f"  {menu:<12} {len(grp):>8,} {w1:>6,} {w2:>6,} {pct:>5.1f}%")

    print(f"\n  รวมทั้งหมด : {len(result_df):,} คู่")

    # ── บันทึก ──
    cols = ["Image 1", "Image 2", "Menu", "Winner", "Num Voter", "Num Vote 1", "Num Vote 2"]
    result_df[cols].to_csv(CONFIG["output_csv"], index=False, encoding="utf-8-sig")
    print(f"  บันทึกที่    : {CONFIG['output_csv']}")
    print(f"\n  ใช้ไฟล์นี้เป็น csv_path_2 หรือ csv_path_3 ใน train_per_menu.py ได้เลยครับ")
    print("=" * 60)


if __name__ == "__main__":
    main()
