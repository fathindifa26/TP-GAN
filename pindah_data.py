import os
import shutil
from PIL import Image

base_dir = 'dataset/MultiPie'
output_dir = 'dataset/MultiPie'

# Folder sumber
frontal_dir = os.path.join(base_dir, 'frontal')
profile_dir = os.path.join(base_dir, 'profile')

# Folder tujuan
resolutions = ['128x128', '64x64', '32x32']
patch_types = ['left_eye', 'right_eye', 'nose', 'mouth']

# Buat folder resolusi dan patch
for res in resolutions:
    os.makedirs(os.path.join(output_dir, res), exist_ok=True)

for p in patch_types:
    os.makedirs(os.path.join(output_dir, 'patch', p), exist_ok=True)


# Fungsi resize
def resize_and_save(img, save_path, size):
    img = img.resize((size, size), Image.LANCZOS)
    img.save(save_path)


# Proses gambar frontal dan profile
for pose_dir in [frontal_dir, profile_dir]:
    for identity in os.listdir(pose_dir):
        identity_path = os.path.join(pose_dir, identity)
        if not os.path.isdir(identity_path): continue

        for fname in os.listdir(identity_path):
            if not fname.endswith('.png'): continue
            src_path = os.path.join(identity_path, fname)
            img = Image.open(src_path)

            # Simpan ke 128x128
            save_128 = os.path.join(output_dir, '128x128', fname)
            img.save(save_128)

            # Simpan ke 64x64 dan 32x32
            resize_and_save(img, os.path.join(output_dir, '64x64', fname), 64)
            resize_and_save(img, os.path.join(output_dir, '32x32', fname), 32)