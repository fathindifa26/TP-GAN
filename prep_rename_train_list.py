import os
from collections import defaultdict

# File input lama
old_frontal = 'frontal_list_old.txt'
old_train = 'train_list_old.txt'

# File output baru
new_frontal = 'frontal_list.txt'
new_train = 'train_list.txt'

POSE_FRONTAL = '051'

# Load isi file lama
with open(old_frontal, 'r') as f:
    frontal_paths = [line.strip() for line in f.readlines()]
with open(old_train, 'r') as f:
    profile_paths = [line.strip() for line in f.readlines()]

# Tracking index per subject per pose
pose_index_counter = defaultdict(lambda: defaultdict(int))
rename_map = {}

def format_new_name(subject_id, pose, index):
    # Format sama persis dengan script rename utama
    return f"{subject_id}_01_01_{pose}_{index:02d}_crop_128.jpg"

# === Rename frontal (pose 051) ===
for path in frontal_paths:
    fname = os.path.basename(path)
    subject_id = fname.split('_')[0]
    idx = pose_index_counter[subject_id][POSE_FRONTAL]
    new_name = format_new_name(subject_id, POSE_FRONTAL, idx)
    new_path = path.replace(fname, new_name)
    rename_map[path] = new_path
    pose_index_counter[subject_id][POSE_FRONTAL] += 1

# === Rename profile (pose ≠ 051) ===
pose_counter = defaultdict(lambda: 1)  # pose number counter
for path in profile_paths:
    fname = os.path.basename(path)
    subject_id = fname.split('_')[0]

    while True:
        pose_id = f"{pose_counter[subject_id]:03d}"
        if pose_id != POSE_FRONTAL:
            break
        pose_counter[subject_id] += 1

    idx = pose_index_counter[subject_id][pose_id]
    new_name = format_new_name(subject_id, pose_id, idx)
    new_path = path.replace(fname, new_name)
    rename_map[path] = new_path
    pose_index_counter[subject_id][pose_id] += 1
    pose_counter[subject_id] += 1

# === Simpan ke file output ===
with open(new_frontal, 'w') as f:
    for path in frontal_paths:
        f.write(rename_map[path] + '\n')

with open(new_train, 'w') as f:
    for path in profile_paths:
        f.write(rename_map[path] + '\n')

print("✅ frontal_list.txt dan train_list.txt berhasil dibuat sesuai format Multi-PIE!")