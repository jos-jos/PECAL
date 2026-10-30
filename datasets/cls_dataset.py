# datasets/cls_dataset.py
import os, cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, img_size=224):
        """
        image_paths: list of image file paths
        labels: list/array of int labels (same length)
        transform: albumentations.Compose
        img_size: default size for error handling (black image)
        """
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图片路径
        img_path = self.image_paths[idx]
        img = cv2.imread(self.image_paths[idx])
        
        # 如果图片读取失败，创建黑色替代图片
        if img is None:
            print(f"警告: 无法读取图片 {img_path}, 使用黑色图片替代")
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        else:
            # 从BGR转换为RGB
            img = img[:, :, ::-1]
            
        label = int(self.labels[idx])
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
        return img, label
