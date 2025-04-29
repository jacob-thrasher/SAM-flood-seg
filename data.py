import os
import torch
import numpy as np
import cv2
import random
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from tifffile import imread

from scipy.ndimage import label

def select_random_points_per_region(mask, connectivity=1, k=1):
    """
    Select one random (row, col) point from each contiguous region (component)
    in a binary segmentation mask where the mask == 1.

    Args:
        mask (np.ndarray): A 2D numpy array with binary values (0 or 1).
        connectivity (int): 1 for 4-connectivity, 2 for 8-connectivity.

    Returns:
        List[tuple]: A list of (row, col) points, one from each region.
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError("Input mask must be a numpy array")
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")

    # Label connected components
    labeled_mask, num_features = label(mask, structure=np.ones((3, 3)) if connectivity == 2 else None)

    points = []
    for region_id in range(1, num_features + 1):
        y_indices, x_indices = np.where(labeled_mask == region_id)
        if len(y_indices) == 0:
            continue  # skip empty region (shouldn't happen)
        for i in range(k):
            idx = random.randint(0, len(y_indices) - 1)
            points.append((x_indices[idx], y_indices[idx]))

    return points

def get_random_point(mask, k=1):
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")

    # Get all indices where mask == 1
    y_indices, x_indices = np.where(mask > 0)

    if len(y_indices) == 0:
        print("No area of interest (1s) found in the mask.")
        return None

    # Choose a random index
    points = []
    for i in range(k):
        idx = random.randint(0, len(y_indices) - 1)
        points.append([x_indices[idx], y_indices[idx]])
    
    return points


def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


class FloodSeg(Dataset):
    def __init__(self, root, df_path, processor, region_select='bbox', k=1, transform=None):
        self.root = root
        self.df = pd.read_csv(df_path)
        self.transform = transform
        self.processor = processor
        self.region_select = region_select
        self.k = k
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.root, 'Image', self.df.iloc[idx]['Image'])).convert('RGB')
        mask = Image.open(os.path.join(self.root, 'Mask', self.df.iloc[idx]['Mask'])).convert('L')

        img = img.resize((256, 256))
        mask = np.array(mask.resize((256, 256)), dtype=np.float32) // 255

        if self.region_select == 'bbox':
            bbox = get_bounding_box(mask)
            inputs = self.processor(img, input_boxes=[[bbox]], return_tensors='pt')
        elif self.region_select == 'point':
            points = get_random_point(mask, self.k)
            inputs = self.processor(img, input_points=[points], return_tensors='pt')
        elif self.region_select == 'each':
            points = select_random_points_per_region(mask, connectivity=1)
            inputs = self.processor(img, input_points=[points], return_tensors='pt')

        # if self.transform:
        #     img = self.transform(img)
        #     mask = self.transform(mask)

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['ground_truth_mask'] = mask
        # inputs['resized_img'] = img

        return inputs

class Sen1Flood11(Dataset):
    def __init__(self, root, df, processor, region_select='bbox', k=1, transform=None):
        self.root = root
        self.df = df
        self.images = os.listdir(os.path.join(root, 'S2Hand'))
        self.masks = os.listdir(os.path.join(self.root, 'LabelHand'))

        self.processor = processor
        self.region_select = region_select
        self.k = k
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'S2Hand', self.df.iloc[idx]['images'])
        mask_path = os.path.join(self.root, 'LabelHand', self.df.iloc[idx]['masks'])

        img_full = imread(img_path)
        mask = imread(mask_path)

        img_rgb = np.stack([
            img_full[3],
            img_full[2],
            img_full[1],
        ])

        img_rgb = img_rgb / img_rgb.max()
        mask = np.maximum(0, mask)

        img = np.reshape(img_rgb, (256, 256))
        mask = np.reshape(mask, (256, 256))

        if self.region_select == 'bbox':
            bbox = get_bounding_box(mask)
            inputs = self.processor(img, input_boxes=[[bbox]], return_tensors='pt')
        elif self.region_select == 'point':
            points = get_random_point(mask, self.k)
            inputs = self.processor(img, input_points=[points], return_tensors='pt')
        elif self.region_select == 'each':
            points = select_random_points_per_region(mask, connectivity=1)
            inputs = self.processor(img, input_points=[points], return_tensors='pt')

        # if self.transform:
        #     img = self.transform(img)
        #     mask = self.transform(mask)

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['ground_truth_mask'] = mask
        return img_rgb, mask