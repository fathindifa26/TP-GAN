import os
from PIL import Image
from tqdm import tqdm

SRC_DIR = "dataset/128x128"           # folder asli 178x218
OUT_DIR = "dataset/128x128"       # folder hasil resize 128x128

os.makedirs(OUT_DIR, exist_ok=True)

for fname in tqdm(sorted(os.listdir(SRC_DIR))):
    if not fname.endswith('.jpg'):
        continue
    img = Image.open(os.path.join(SRC_DIR, fname)).convert("RGB")
    img = img.resize((128, 128), Image.LANCZOS)
    img.save(os.path.join(OUT_DIR, fname))

print("✅ Resize selesai ke 128x128.")
