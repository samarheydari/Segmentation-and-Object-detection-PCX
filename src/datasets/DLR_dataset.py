import torch
import numpy as np
from torch.utils.data import Dataset
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted

import sys
sys.path.append("..")
from src.utils_DLR import tile_array

class DatasetDLR(Dataset):
    class_names = ["flood"]

    def __init__(self, img_dir, mask_dir=None, input_bands_idx=[0, 1, 2, 3], 
                 normalize_means_stds=None, tile_size=256, overlap=0.2, padding=True, only_RGB=False):
        """
        Args:
            img_dir (str or Path): Path to the directory containing image .tif files.
            mask_dir (str or Path, optional): Path to the directory containing mask .tif files.
            input_bands_idx (list): Indices of bands to keep (default: [0,1,2,3]).
            normalize_means_stds (tuple, optional): Tuple (means, stds) for normalization.
            tile_size (int): Size of tiles to create from images (only if larger than tile_size).
            overlap (float): Overlap between tiles.
            padding (bool): Whether to pad tiles if necessary.
        """
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.input_bands_idx = input_bands_idx
        self.normalize_means_stds = normalize_means_stds
        self.tile_size = tile_size
        self.overlap = overlap
        self.padding = padding
        self.only_RGB = only_RGB

        self.img_files = natsorted(list(self.img_dir.glob("*.tif*")), key=str)
        self.mask_files = natsorted(list(self.mask_dir.glob("*.tif*")), key=str) if self.mask_dir else None

        self.img_arr, self.msk_arr = self.load_data()

        self.reverse_normalization = torch.nn.Identity()

    def tile_if_needed(self, img):
        """Tile image if its size is larger than the tile size."""
        if img.shape[0] > self.tile_size or img.shape[1] > self.tile_size:
            return tile_array(img, xsize=self.tile_size, ysize=self.tile_size, 
                              overlap=self.overlap, padding=self.padding)
        else:
            return np.expand_dims(img, axis=0)  # Keep original shape, add batch dim

    def load_data(self):
        """Loads and tiles images and masks if needed."""
        img_list, mask_list = [], []
        
        for img_file in tqdm(self.img_files, desc="Loading images"):
            img = tiff.imread(img_file)
            img_tiled = self.tile_if_needed(img)
            img_list.append(img_tiled)
        
        img_arr = np.concatenate(img_list, axis=0)  # Shape (N, H, W, C)
        img_arr = img_arr[:, :, :, self.input_bands_idx].astype(np.float32) / 10000.0

        if self.mask_files:
            for mask_file in tqdm(self.mask_files, desc="Loading masks"):
                mask = tiff.imread(mask_file)
                mask_tiled = self.tile_if_needed(mask)
                mask_list.append(mask_tiled)
            
            msk_arr = np.concatenate(mask_list, axis=0).astype(np.int64)  # Shape (N, H, W)
        else:
            msk_arr = None

        if self.only_RGB:
            img_arr = img_arr[:, :, :, :3]  # Keep only first 3 channels (RGB)
        return img_arr, msk_arr

    def normalize_img(self, img, means, stds):
        bands = []
        if self.only_RGB:
            for i in range(3):
                bands.append((img[:, :, i] - means[i]) / stds[i])
        else:
            for i in range(len(self.input_bands_idx)):
                bands.append((img[:, :, i] - means[i]) / stds[i])
        return np.dstack(bands)

    def __getitem__(self, idx):
        img = self.img_arr[idx]
        if self.normalize_means_stds:
            img = self.normalize_img(img, self.normalize_means_stds[0], self.normalize_means_stds[1])

        img = img.transpose(2, 0, 1).astype(np.float32)  # Convert to (C, H, W)

        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img)

        if self.msk_arr is not None:
            mask = self.msk_arr[idx]  # (H, W)
            mask_tensor = torch.from_numpy(mask).long()
            return img_tensor, mask_tensor
        else:
            return img_tensor, None

    def __len__(self):
        return self.img_arr.shape[0]
    
    def reverse_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        return torch.multiply(((data - data.min()) / (data.max() - data.min())), 255).type(torch.uint8).detach().cpu()
