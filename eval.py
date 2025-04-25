import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from segment_anything import SamPredictor, sam_model_registry
from transformers import SamProcessor, SamModel
from torch.optim import Adam
from statistics import mean
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
import monai
from data import FloodSeg

def get_metrics(model, dataloader):

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    model.eval()
    running_dice = 0
    with torch.no_grad():
        for batch in dataloader:

            # Configure inputs
            inputs = {k: v.to(device) for k, v in batch.items() if k != "ground_truth_mask"}

            # Get masks
            outputs = model(**inputs, multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)

            # Calculate loss
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            running_dice += (1 - loss)

    print(f'Dice: {running_dice / len(dataloader)}')


root = '/home/WVU-AD/jdt0025/Documents/data/Flood'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load processor and model
## Processor preprocesses images in the way that SAM expects them to be
processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
model = SamModel.from_pretrained('facebook/sam-vit-base')


# Define dataloaders
train_dataset = FloodSeg(root, os.path.join(root, 'train.csv'), processor, region_select='point', k=1)
test_dataset = FloodSeg(root, os.path.join(root, 'test.csv'), processor, region_select='point', k=1)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

model.load_state_dict(torch.load('experiments/best_model_SAM_1.pth', weights_only=True))
model.to('cuda')


get_metrics(model, test_dataloader)

batch = next(iter(test_dataloader))
inputs = {k: v.to(device) for k, v in batch.items() if k != "ground_truth_mask"}
outputs = model(**inputs, multimask_output=False)
pred = outputs.pred_masks.squeeze().detach().cpu()


# Get segmentation mask for test imafge

# Get test image and mask
img = batch['pixel_values'][0].detach().cpu()
mask = batch['ground_truth_mask'][0].detach().cpu()
resize = T.Resize((256, 256))

# Normalize image but I don't remember why lol
A = img
A -= A.min(1, keepdim=True)[0]
A /= A.max(1, keepdim=True)[0]
img = resize(A)
mask = resize(mask)


# Convert unbounded predictions into probabilities (sigmoid)
# Round probabilities to nearest value (0, 1)
test = torch.round(F.sigmoid(pred[0]))
print(test)

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(img.permute(1, 2, 0))
ax[0].set_title('Input Image')
ax[1].imshow(mask, alpha=1)
ax[1].set_title('Ground Truth Mask')
ax[2].imshow(test, alpha=1)
ax[2].set_title('Predicted Mask (Single point)')

plt.show()
print(pred.size())



