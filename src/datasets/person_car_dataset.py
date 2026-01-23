import os
from torch.utils.data import Dataset
from PIL import Image
import natsort

import torch
from torchvision import transforms as T

class PersonCarDataset(Dataset):
    class_names = ("person", "car") 

    def __init__(self, root_dir, split="train", transform=None):
        self.image_dir = os.path.join(root_dir, "images", split)
        self.label_dir = os.path.join(root_dir, "labels", split)
        self.transform = transform

        # Get list of image and label files, ensuring they are sorted correctly
        self.image_files = natsort.natsorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))])# Updated the hardcoded '.JPG' to support additional image formats
        self.label_files = natsort.natsorted([f for f in os.listdir(self.label_dir) if f.endswith(".txt")])

        # Filter out labels that don't have images
        image_names = set([f.split(".")[0] for f in self.image_files])
        self.label_files = [f for f in self.label_files if f.split(".")[0] in image_names]

        # Same for images
        label_names = set([f.split(".")[0] for f in self.label_files])
        self.image_files = [f for f in self.image_files if f.split(".")[0] in label_names]

        # Ensure the number of images and masks match
        assert len(self.image_files) == len(self.label_files), "Mismatch between number of images and masks!"

        # Ensure image names and label names match
        for image_file, label_file in zip(self.image_files, self.label_files):
            assert image_file.split(".")[0] == label_file.split(".")[0], f"Mismatch between image and label files: {image_file} and {label_file}"
    
    def reverse_normalization(self, data: torch.Tensor) -> torch.Tensor:
        return torch.multiply(data, 255).detach().cpu()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label_file = self.label_files[idx]


        img_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, label_file)

        image = Image.open(img_path).convert("RGB")  # Load RGB image
        with open(label_path, "r") as f:
            objects_lines = f.readlines()
        
        objects = [self.convert_label_line_to_dict(line) for line in objects_lines]

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        labels = [obj["class_id"] for obj in objects] 
        targets = torch.tensor(labels)
        # Convert target vector to shape [num_objects, 2] by adding a dimension and duplicating values
        # This ensures each object's class ID is represented as [class_id, class_id] to match the format in coco dataset in L-CRP code
        targets_transformed = targets[:, None].expand(targets.shape[0], 2)

        return image, targets_transformed
    
    def convert_label_line_to_dict(self, line: str) -> dict:
        class_id, x_center, y_center, width, height = line.split(" ")
        return {
            "class_id": int(class_id),
            "x_center": float(x_center),
            "y_center": float(y_center),
            "width": float(width),
            "height": float(height)
        }
