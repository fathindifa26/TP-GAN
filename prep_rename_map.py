import os
from tqdm import tqdm

identity_path = "identity_CelebA.txt"         # file dari CelebA
img_dir = "dataset/datasets/img_align_celeba"  # folder asli hasil ekstrak
output_dir = "dataset/128x128"                # folder output hasil rename

os.makedirs(output_dir, exist_ok=True)

with open(identity_path, 'r') as f:
    lines = f.readlines()

# {identity_id: [filename1, filename2, ...]}
id_dict = {}
for line in lines:
    fname, identity = line.strip().split()
    if identity not in id_dict:
        id_dict[identity] = []
    id_dict[identity].append(fname)

# Rename dan simpan
index = 0
img_mapping = {}  # {new_filename: identity}
for identity, files in tqdm(id_dict.items()):
    files = sorted(files)  # urutkan biar konsisten
    for i, fname in enumerate(files):
        new_name = f"{int(identity):05d}_{i:03d}.jpg"
        src = os.path.join(img_dir, fname)
        dst = os.path.join(output_dir, new_name)
        if os.path.exists(src):
            os.rename(src, dst)
            img_mapping[new_name] = int(identity)

print(f"✅ Rename selesai. Total: {len(img_mapping)} file.")
