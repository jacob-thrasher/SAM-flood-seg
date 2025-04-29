import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from random import shuffle
from segment_anything import SamPredictor, sam_model_registry
from data import FloodSeg, Sen1Flood11
from transformers import SamProcessor, SamModel
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd


root = '/home/WVU-AD/jdt0025/Documents/data/v1.1/data/flood_events/HandLabeled'


# dataset = Sen1Flood11(root)
# img, mask = dataset[4]

# print(img.shape)
# print(mask.shape)



# fig, ax = plt.subplots(nrows=1, ncols=2)

# img = torch.tensor(img)
# ax[0].imshow(img.permute(1, 2, 0))
# ax[1].imshow(mask, alpha=1)
# plt.show()


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
# model = SamModel.from_pretrained('facebook/sam-vit-base')

# train_dataset = FloodSeg(os.path.join(root, 'Train/Image'), os.path.join(root, 'Train/Mask'), processor, region_select='each', k=1)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


# inputs = train_dataset[2]
# # bbox = inputs['input_boxes'][0]
# # bbox = [x // 4 for x in bbox]
# # print(inputs['pixel_values'].size())
# # print(inputs['ground_truth_mask'].shape)

# points = inputs['input_points']
# # points = [x // 4 for x in points]

# fig, ax = plt.subplots(nrows=1, ncols=2)
# A = inputs['pixel_values']
# A -= A.min(1, keepdim=True)[0]
# A /= A.max(1, keepdim=True)[0]
# ax[0].imshow(A.permute(1, 2, 0))
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# ax[1].imshow(inputs['ground_truth_mask'], alpha=1)

# # rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], edgecolor='red', linewidth=2, facecolor='none')
# # ax[1].add_patch(rect)

# points = points.squeeze(0)
# print(points)
# x = points[:, 0] // 4
# y = points[:, 1] // 4
# print(x)
# print(y)

# ax[1].scatter(x, y, color='red', marker='^', s=100)


# ax[1].set_xticks([])
# ax[1].set_yticks([])
# plt.show()