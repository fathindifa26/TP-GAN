import os
from collections import defaultdict

# Konfigurasi
frontal_list_path = 'frontal_list_old.txt'
train_list_path = 'train_list_old.txt'
POSE_FRONTAL = '051'

SUBFOLDERS = ['128x128', '64x64', '32x32']
PATCH_PARTS = ['left_eye', 'right_eye', 'nose', 'mouth']
BASE_DIR = 'dataset'

# Ambil daftar file
with open(frontal_list_path, 'r') as f:
    frontal_paths = [line.strip() for line in f.readlines()]
with open(train_list_path, 'r') as f:
    profile_paths = [line.strip() for line in f.readlines()]

# Tracking index per subject per pose
pose_index_counter = defaultdict(lambda: defaultdict(int))
rename_map = {}  # old filename (.jpg) -> new filename (.jpg)

def format_new_name(subject_id, pose, index):
    return f"{subject_id}_01_01_{pose}_{index:02d}_crop_128.jpg"

# Proses frontal
for path in frontal_paths:
    fname = os.path.basename(path)
    subject_id = fname.split('_')[0]
    index = pose_index_counter[subject_id][POSE_FRONTAL]
    new_fname = format_new_name(subject_id, POSE_FRONTAL, index)
    rename_map[fname] = new_fname
    pose_index_counter[subject_id][POSE_FRONTAL] += 1

# Proses profile
pose_counter = defaultdict(lambda: 1)
for path in profile_paths:
    fname = os.path.basename(path)
    subject_id = fname.split('_')[0]
    while True:
        pose_id = f"{pose_counter[subject_id]:03d}"
        if pose_id != POSE_FRONTAL:
            break
        pose_counter[subject_id] += 1
    index = pose_index_counter[subject_id][pose_id]
    new_fname = format_new_name(subject_id, pose_id, index)
    rename_map[fname] = new_fname
    pose_index_counter[subject_id][pose_id] += 1
    pose_counter[subject_id] += 1

# Helper untuk rename file dalam folder
def rename_in_folder(folder_path, is_landmark=False):
    for old_name, new_name in rename_map.items():
        old_base = os.path.splitext(old_name)[0]
        new_base = os.path.splitext(new_name)[0]
        ext = '.txt' if is_landmark else '.jpg'
        old_path = os.path.join(folder_path, old_base + ext)
        new_path = os.path.join(folder_path, new_base + ext)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"✅ {old_path} → {new_path}")
        else:
            print(f"⚠️ File not found: {old_path}")

# Rename di subfolder 128x128, 64x64, 32x32
for sub in SUBFOLDERS:
    print(f"\n📁 Proses: {sub}")
    rename_in_folder(os.path.join(BASE_DIR, sub))

# Rename landmark .txt
print(f"\n📁 Proses: landmarks/")
rename_in_folder(os.path.join(BASE_DIR, 'landmarks'), is_landmark=True)

# Rename semua patch
for patch in PATCH_PARTS:
    print(f"\n📁 Proses: patch/{patch}/")
    rename_in_folder(os.path.join(BASE_DIR, 'patch', patch))

print("\n🎉 Selesai rename semua file termasuk landmark (.txt) dan patch!")
