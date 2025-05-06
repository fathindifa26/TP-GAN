import os
import numpy as np
from PIL import Image
import face_alignment

IMG_DIR = 'dataset/organized/128x128'  # sesuaikan
LM_DIR = 'dataset/landmarks'
os.makedirs(LM_DIR, exist_ok=True)

fa = face_alignment.FaceAlignment('2D', flip_input=False)

img_list = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])

for img_name in img_list:
    img_path = os.path.join(IMG_DIR, img_name)
    img = np.array(Image.open(img_path).convert('RGB'))

    preds = fa.get_landmarks(img)

    if preds is None:
        print(f"[!] Landmark not detected: {img_name}")
        continue

    lm = preds[0]  # ambil wajah pertama jika ada lebih dari satu
    save_path = os.path.join(LM_DIR, img_name.replace('.png', '.txt'))
    np.savetxt(save_path, lm, fmt='%.6f')

print("✅ Landmark .txt file generation complete.")
