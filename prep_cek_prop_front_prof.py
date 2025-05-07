from collections import defaultdict
import os

def extract_identity(path):
    fname = os.path.basename(path)
    identity = fname.split('_')[0]
    return identity

# Baca file
with open("frontal_list_old.txt") as f:
    frontal_files = [line.strip() for line in f.readlines()]

with open("train_list_old.txt") as f:
    profile_files = [line.strip() for line in f.readlines()]

# Hitung jumlah per identitas
count_dict = defaultdict(lambda: {'frontal': 0, 'profile': 0})

for f in frontal_files:
    id_ = extract_identity(f)
    count_dict[id_]['frontal'] += 1

for f in profile_files:
    id_ = extract_identity(f)
    count_dict[id_]['profile'] += 1

# Tampilkan hasil
print(f"{'ID':<10} {'Frontal':>8} {'Profile':>8} {'Total':>8} {'Frontal%':>10} {'Profile%':>10}")
print('-' * 60)
for id_, counts in sorted(count_dict.items()):
    total = counts['frontal'] + counts['profile']
    f_pct = counts['frontal'] / total * 100 if total else 0
    p_pct = counts['profile'] / total * 100 if total else 0
    print(f"{id_:<10} {counts['frontal']:>8} {counts['profile']:>8} {total:>8} {f'{f_pct:.1f}%':>10} {f'{p_pct:.1f}%':>10}")
