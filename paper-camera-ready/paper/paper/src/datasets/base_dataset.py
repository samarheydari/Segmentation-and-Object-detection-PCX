# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import random

from torch.nn import functional as F
from torch.utils import data

y_k_size = 6
x_k_size = 6




class BaseDataset(data.Dataset):
    def __init__(self,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor

        self.files = []

    def __len__(self):
        return len(self.files)
    

    def input_transform(self, image, city=True):
        if city:
            image = image.astype(np.float32)[:, :, ::-1]
        else:
            image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        # keep sentinel values such as -1 intact for torch's ignore_index
        return np.array(label).astype(np.int64)

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label, edge):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))
        edge = self.pad_image(edge, h, w, self.crop_size,
                               (0.0,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        edge = edge[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label, edge

    def multi_scale_aug(self, image, label=None, edge=None,
                        rand_scale=1, rand_crop=True):
        long_size = np.int_(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int_(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int_(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
            if edge is not None:
                edge = cv2.resize(edge, (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label, edge = self.rand_crop(image, label, edge)

        return image, label, edge


    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True, edge_pad=True, edge_size=4, city=True):
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label_arr = label

        # Build a uint8 copy only for Canny edge detection
        label_for_edge = label_arr
        if label_for_edge.dtype != np.uint8:
            label_for_edge = label_for_edge.astype(np.uint8)

        if label_arr.ndim == 3:
            try:
                label_gray = cv2.cvtColor(label_for_edge, cv2.COLOR_RGB2GRAY)
            except Exception:
                label_gray = label_for_edge[..., 0]
        else:
            label_gray = label_for_edge

        edge = cv2.Canny(label_gray, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0
        
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label_arr, edge = self.multi_scale_aug(image, label_arr, edge,
                                                rand_scale=rand_scale)

        image = self.input_transform(image, city=city)
        label = self.label_transform(label_arr)
        

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            edge = edge[:, ::flip]
        if image.shape[1] != self.crop_size[0] or image.shape[2] != self.crop_size[1]:
            image = cv2.resize(image.transpose(1,2,0), (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
            label = cv2.resize(label, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_NEAREST)
            edge = cv2.resize(edge, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_NEAREST)
        return image, label, edge



    def inference(self, config, model, image):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]
        
        
        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )
        # print(pred)

        # pred_soft = F.softmax(pred, dim=1)
        # print(pred_soft.exp().shape)
        
        return pred.exp()
