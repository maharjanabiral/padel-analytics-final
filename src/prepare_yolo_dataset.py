import os
import shutil
from pathlib import Path
 
# ── Configure these paths ──────────────────────────────
BALL_DIR   = "ball_dataset"   # has train / val / test
RACKET_DIR = "racket_dataset"        # has train / val
OUTPUT_DIR = "merged_dataset"
 
# Class index used in each source dataset's label files
BALL_SOURCE_CLASS   = 0   # → will become class 0 in merged
RACKET_SOURCE_CLASS = 0   # → will become class 1 in merged
# ──────────────────────────────────────────────────────
 
 
def copy_files(src_root, split, src_class, dst_class, prefix):
    src_images = Path(src_root) / split / "images"
    src_labels = Path(src_root) / split / "labels"
    dst_images = Path(OUTPUT_DIR) / split / "images"
    dst_labels = Path(OUTPUT_DIR) / split / "labels"
 
    if not src_images.exists():
        return 0
 
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
 
    count = 0
    for img in src_images.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
 
        # Copy image with prefix to avoid name collisions
        shutil.copy2(img, dst_images / f"{prefix}_{img.name}")
 
        # Remap label class and copy
        lbl_src = src_labels / (img.stem + ".txt")
        lbl_dst = dst_labels / f"{prefix}_{img.stem}.txt"
 
        if lbl_src.exists():
            lines = []
            for line in lbl_src.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls = dst_class if int(parts[0]) == src_class else int(parts[0])
                lines.append(f"{cls} {' '.join(parts[1:])}")
            lbl_dst.write_text("\n".join(lines))
        else:
            lbl_dst.write_text("")
 
        count += 1
 
    print(f"  {src_root}/{split} -> {split}/  ({count} images)")
    return count
 
 
# ── Run ───────────────────────────────────────────────
print("Merging datasets...\n")
 
for split in ("train", "val", "test"):
    copy_files(BALL_DIR,   split, BALL_SOURCE_CLASS,   dst_class=0, prefix="ball")
 
for split in ("train", "val"):
    copy_files(RACKET_DIR, split, RACKET_SOURCE_CLASS, dst_class=1, prefix="racket")