import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import face_alignment
import torchvision.transforms as transforms

# ====== Konfigurasi ======
IMG_DIR = 'dataset/128x128'          # Gambar asli (hasil resize 128x128)
LM_DIR = 'dataset/landmarks'         # Simpan landmark .txt
PATCH_DIR = 'dataset/patch'          # Simpan hasil crop patch
OUT64_DIR = 'dataset/64x64'          # Simpan resize 64x64
OUT32_DIR = 'dataset/32x32'          # Simpan resize 32x32
os.makedirs(LM_DIR, exist_ok=True)
os.makedirs(OUT64_DIR, exist_ok=True)
os.makedirs(OUT32_DIR, exist_ok=True)
for part in ['left_eye', 'right_eye', 'nose', 'mouth']:
    os.makedirs(os.path.join(PATCH_DIR, part), exist_ok=True)

# ====== Landmark Detector ======
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')


# ====== Patch Size Config ======
PATCH_SIZE = {
    'left_eye':  (40, 40),
    'right_eye': (40, 40),
    'nose':      (40, 32),
    'mouth':     (48, 32),
}

# ====== Loop Semua Gambar ======
image_list = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
for fname in tqdm(image_list):
    img_path = os.path.join(IMG_DIR, fname)
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)

    # ====== Landmark Detection ======
    try:
        landmarks = fa.get_landmarks(img_np)
    except:
        print(f"[!] Error reading: {fname}")
        continue

    if landmarks is None:
        print(f"[!] Landmark not found: {fname}")
        continue

    landmark = landmarks[0]
    np.savetxt(os.path.join(LM_DIR, fname.replace('.jpg', '.txt')), landmark, fmt='%.6f')

    # ====== Save 64x64 and 32x32 ======
    img.resize((64, 64), Image.LANCZOS).save(os.path.join(OUT64_DIR, fname))
    img.resize((32, 32), Image.LANCZOS).save(os.path.join(OUT32_DIR, fname))

    # ====== Crop Patch ======
    # Landmark index referensi (dari 68 points):
    le = landmark[36:42].mean(axis=0)
    re = landmark[42:48].mean(axis=0)
    no = landmark[27:36].mean(axis=0)
    mo = landmark[48:68].mean(axis=0)

    centers = {
        'left_eye': le,
        'right_eye': re,
        'nose': no,
        'mouth': mo
    }

    for part in centers:
        cx, cy = centers[part]
        w, h = PATCH_SIZE[part]
        x1 = int(cx - w // 2)
        y1 = int(cy - h // 2)
        x2 = x1 + w
        y2 = y1 + h

        # Crop dan simpan
        cropped = img.crop((x1, y1, x2, y2))
        patch_path = os.path.join(PATCH_DIR, part, fname)
        cropped.save(patch_path)

print("✅ Dataset preparation selesai.")
