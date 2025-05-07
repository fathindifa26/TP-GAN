import os
import math
from tqdm import tqdm
import face_alignment
from PIL import Image
import numpy as np

IMG_DIR = 'dataset/128x128'
TRAIN_LIST_PATH = 'train_list_old.txt'
FRONTAL_LIST_PATH = 'frontal_list_old.txt'

# Threshold konfigurasi
ANGLE_THRESHOLD = 3
EYE_Y_DIFF_THRESHOLD = 1.0
NOSE_OFFSET_THRESHOLD = 2.0

# Init face alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

train_list = []
frontal_list = []

print("🔍 Memproses gambar...")
for fname in tqdm(sorted(os.listdir(IMG_DIR))):
    if not fname.endswith('.jpg'):
        continue

    path = os.path.join(IMG_DIR, fname)
    img = Image.open(path).convert('RGB')
    np_img = np.array(img)

    try:
        landmarks = fa.get_landmarks(np_img)
    except:
        print(f"[!] Error reading: {fname}")
        continue

    if landmarks is None:
        print(f"[!] Landmark not found: {fname}")
        continue

    lm = landmarks[0]
    le = lm[36:42].mean(axis=0)
    re = lm[42:48].mean(axis=0)
    nose = lm[27:36].mean(axis=0)

    # hitung sudut antara mata
    dx = re[0] - le[0]
    dy = re[1] - le[1]
    angle = abs(math.degrees(math.atan2(dy, dx)))

    # beda ketinggian antara mata
    eye_y_diff = abs(re[1] - le[1])

    # posisi hidung relatif terhadap tengah-tengah mata
    mid_eye_x = (le[0] + re[0]) / 2
    nose_offset = abs(nose[0] - mid_eye_x)

    # cetak informasi debug
    print(f"{fname:20s} | angle: {angle:5.1f}° | eye_y_diff: {eye_y_diff:5.1f} | nose_offset: {nose_offset:5.1f}")

    # gunakan kombinasi 3 kriteria
    is_profile = (
        angle > ANGLE_THRESHOLD or
        eye_y_diff > EYE_Y_DIFF_THRESHOLD or
        nose_offset > NOSE_OFFSET_THRESHOLD
    )

    if is_profile:
        train_list.append(os.path.join(IMG_DIR, fname))
    else:
        frontal_list.append(os.path.join(IMG_DIR, fname))

# Simpan file
with open(TRAIN_LIST_PATH, 'w') as f:
    for p in train_list:
        f.write(p + '\n')

with open(FRONTAL_LIST_PATH, 'w') as f:
    for p in frontal_list:
        f.write(p + '\n')

print(f"✅ Selesai! Jumlah frontal: {len(frontal_list)}, profile: {len(train_list)}")
