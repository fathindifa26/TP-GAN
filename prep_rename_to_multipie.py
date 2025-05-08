import os
import shutil
from collections import defaultdict

# ============ KONFIGURASI ============
FRONTAL_LIST_OLD = 'frontal_list_old.txt'
TRAIN_LIST_OLD   = 'train_list_old.txt'
FRONTAL_LIST_NEW = 'frontal_list.txt'
TRAIN_LIST_NEW   = 'train_list.txt'

BASE_DIR     = 'dataset'
SUBFOLDERS   = ['128x128', '64x64', '32x32']
LANDMARK_DIR = os.path.join(BASE_DIR, 'landmarks')
PATCH_DIR    = os.path.join(BASE_DIR, 'patch')
PATCH_PARTS  = ['left_eye', 'right_eye', 'nose', 'mouth']
POSE_FRONTAL = '051'

# ============ STEP 1: Baca daftar path lama ============
def normalize(path):
    return path.strip().replace('\\', '/')

with open(FRONTAL_LIST_OLD, 'r') as f:
    frontal_old = [normalize(line) for line in f if line.strip()]
with open(TRAIN_LIST_OLD, 'r') as f:
    train_old = [normalize(line) for line in f if line.strip()]

# ============ STEP 2: Buat mapping nama baru ============
rename_map = {}
counters = defaultdict(lambda: defaultdict(int))  # {subject: {pose: counter}}

# Proses frontal
for path in frontal_old:
    old_base = os.path.splitext(os.path.basename(path))[0]
    subject = old_base.split('_')[0]
    idx = counters[subject][POSE_FRONTAL]
    new_base = f"{subject}_{POSE_FRONTAL}_{idx:02d}"
    rename_map[old_base] = new_base
    counters[subject][POSE_FRONTAL] += 1

# Proses profile
for path in train_old:
    old_base = os.path.splitext(os.path.basename(path))[0]
    subject, pose = old_base.split('_')
    idx = counters[subject][pose]
    new_base = f"{subject}_{pose}_{idx:02d}"
    rename_map[old_base] = new_base
    counters[subject][pose] += 1

# ============ STEP 3: Rename semua file fisik ============
def rename_files_in_folder(folder, ext):
    for old_base, new_base in rename_map.items():
        old_path = os.path.join(folder, old_base + ext)
        new_path = os.path.join(folder, new_base + ext)
        if os.path.exists(old_path):
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"✅ Rename: {old_path} → {new_path}")
            else:
                print(f"⚠️ Sudah ada: {new_path} (lewati)")
        else:
            print(f"⚠️ Tidak ditemukan: {old_path}")

for sub in SUBFOLDERS:
    print(f"\n📂 Rename di folder {sub}")
    rename_files_in_folder(os.path.join(BASE_DIR, sub), '.jpg')

print(f"\n📂 Rename landmarks (.txt)")
rename_files_in_folder(LANDMARK_DIR, '.txt')

print(f"\n📂 Rename patches")
for part in PATCH_PARTS:
    rename_files_in_folder(os.path.join(PATCH_DIR, part), '.jpg')

# ============ STEP 4: Cek frontal vs profile (REAL from FOLDER) ============
print("\n🔍 Membaca isi folder 128x128 untuk menghitung frontal & profile...")

frontal_by_subject = defaultdict(list)
profile_by_subject = defaultdict(list)

for fname in os.listdir(os.path.join(BASE_DIR, '128x128')):
    if not fname.endswith('.jpg'):
        continue
    base = fname.replace('.jpg', '')
    parts = base.split('_')
    if len(parts) == 3:
        subject, pose, _ = parts
        if pose == POSE_FRONTAL:
            frontal_by_subject[subject].append(base)
        else:
            profile_by_subject[subject].append(base)

# ============ STEP 5: Duplikasi frontal jika perlu ============
print("\n🔁 Cek keseimbangan frontal vs profile:")
for subject in profile_by_subject:
    num_p = len(profile_by_subject[subject])
    num_f = len(frontal_by_subject[subject])
    if num_f < num_p:
        needed = num_p - num_f
        print(f"✂️  Subject {subject} kekurangan {needed} frontal")
        for i in range(needed):
            src_base = frontal_by_subject[subject][i % num_f]
            new_idx = counters[subject][POSE_FRONTAL]
            new_base = f"{subject}_{POSE_FRONTAL}_{new_idx:02d}"
            counters[subject][POSE_FRONTAL] += 1

            for sub in SUBFOLDERS:
                src = os.path.join(BASE_DIR, sub, src_base + '.jpg')
                dst = os.path.join(BASE_DIR, sub, new_base + '.jpg')
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"📑 Copy {src} → {dst}")

            for part in PATCH_PARTS:
                src = os.path.join(PATCH_DIR, part, src_base + '.jpg')
                dst = os.path.join(PATCH_DIR, part, new_base + '.jpg')
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)

            src = os.path.join(LANDMARK_DIR, src_base + '.txt')
            dst = os.path.join(LANDMARK_DIR, new_base + '.txt')
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

            frontal_by_subject[subject].append(new_base)
    else:
        print(f"✔️  Subject {subject} sudah seimbang (frontal={num_f}, profile={num_p})")

# ============ STEP 6: Tulis ulang frontal_list.txt dan train_list.txt ============
print("\n📝 Menulis ulang frontal_list.txt dan train_list.txt...")

with open(FRONTAL_LIST_NEW, 'w') as f:
    for subject in sorted(frontal_by_subject):
        for base in sorted(frontal_by_subject[subject], key=lambda x: int(x.split('_')[-1])):
            path = os.path.join(BASE_DIR, '128x128', base + '.jpg')
            f.write(path.replace('\\', '/') + '\n')

with open(TRAIN_LIST_NEW, 'w') as f:
    for subject in sorted(profile_by_subject):
        for base in sorted(profile_by_subject[subject], key=lambda x: int(x.split('_')[-1])):
            path = os.path.join(BASE_DIR, '128x128', base + '.jpg')
            f.write(path.replace('\\', '/') + '\n')

print("\n✅ Selesai! Semua file sudah di-rename, diduplikasi jika perlu, dan path ditulis dengan '/' ✅")
