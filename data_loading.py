import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
from PIL import Image


class LungSegmentationData(Dataset):
    def __init__(self, image_paths, mask_paths, transform):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("L")
        mask = Image.open(self.mask_paths[item]).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        mask = (mask > 0).float()

        return image, mask


def get_loaders(image_dir, mask_dir, batch_size, transform, val_split=0.2):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = [f.replace('.png', '_mask.png') for f in image_files]
    image_paths = []
    mask_paths = []
    for img in image_files:
        mask_file = img.replace('.png', '_mask.png')
        if not os.path.exists(os.path.join(mask_dir, mask_file)):
            continue
        image_paths.append(os.path.join(image_dir, img))
        mask_paths.append(os.path.join(mask_dir, mask_file))

    #image_paths = [os.path.join(image_dir, img) for img in image_files]
    #mask_paths = [os.path.join(mask_dir, img) for img in mask_files]

    paired = list(zip(image_paths, mask_paths))
    train_pairs, val_pairs = train_test_split(paired, test_size=val_split)

    train_img, train_mask = zip(*train_pairs)
    val_img, val_mask = zip(*val_pairs)

    train_dataset = LungSegmentationData(train_img, train_mask, transform)
    val_dataset = LungSegmentationData(val_img, val_mask, transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)

    return train_loader, val_loader
