# test_data_loader.py
from data.data import TrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

# Load daftar path dari train_list_old.txt
with open('train_list.txt', 'r') as f:
    img_list = [line.strip() for line in f.readlines()]

# Buat dataset dan dataloader
dataset = TrainDataset(img_list)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Ambil satu batch
batch = next(iter(loader))

print("✅ Berhasil memuat 1 batch!")
print("Isi batch:")
for key in batch:
    if isinstance(batch[key], torch.Tensor):
        print(f" - {key}: shape={batch[key].shape}, min={batch[key].min():.2f}, max={batch[key].max():.2f}")
    else:
        print(f" - {key}: {type(batch[key])}")

# Opsional: tampilkan gambar input dan frontal
def imshow(tensor, title):
    img = (tensor[0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0  # Convert to [0,1] range
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(8, 4))
imshow(batch['img'], 'Input Image')
plt.show()
