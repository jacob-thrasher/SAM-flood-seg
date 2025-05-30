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
import monai
import argparse
import random
from data import FloodSeg, Sen1Flood11



parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69, required=False, help="seed")
parser.add_argument("--k", type=int, default=1, required=False, help="Number of points")
parser.add_argument("--region_select", type=str, default='bbox', required=False, help="Region method")
parser.add_argument("--dataset", type=str, default='floodseg', required=False, help="Train dataset")
parser.add_argument("--exp_root", type=str, default='experiments', required=False, help="Path to experiments")
parser.add_argument("--exist_ok", type=int, default=0, required=False, help="Overwrite existing save?")
arguments = parser.parse_args()

seed = arguments.seed
k = arguments.k
region_select = arguments.region_select
dataset = arguments.dataset
exp_root = arguments.exp_root
exp_id = f'{dataset}_{region_select}_k{k}_{seed}'

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if not os.path.exists(os.path.join(exp_root, exp_id)):
    os.mkdir(os.path.join(exp_root, exp_id))
else:
    if arguments.exist_ok == 0:
        raise OSError("Path exists!")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
model = SamModel.from_pretrained('facebook/sam-vit-base')


if dataset == 'floodseg':
    root = '/home/WVU-AD/jdt0025/Documents/data/Flood'
    train_dataset = FloodSeg(root, os.path.join(root, 'train.csv'), processor, region_select=region_select, k=k)
    test_dataset = FloodSeg(root, os.path.join(root, 'test.csv'), processor, region_select=region_select, k=k)
elif dataset == 'sen1flood11':
    root = '/home/WVU-AD/jdt0025/Documents/data/v1.1/data/flood_events/HandLabeled'
    train_dataset = Sen1Flood11(root, os.path.join(root, 'train_cleaner.csv'), processor, region_select=region_select, k=k)
    test_dataset  = Sen1Flood11(root, os.path.join(root, 'test_cleaner.csv'), processor, region_select=region_select, k=k)
    print('Train elements:', len(train_dataset))
    print('Test elements :', len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

best_val_loss = float('inf')
best_model_state = None
metrics = {
    'train_loss': [],
    'valid_loss': []
}
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
        model.train()

        # Forward pass
        inputs = {k: v.to(device) for k, v in batch.items() if k != "ground_truth_mask"}
        outputs = model(**inputs, multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)
        predicted_masks = F.sigmoid(predicted_masks)

        # Calculate loss
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    mean_train_loss = mean(epoch_losses)
    epoch_losses.append(mean_train_loss)
    metrics['train_loss'].append(mean_train_loss)

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')



    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "ground_truth_mask"}
            outputs = model(**inputs, multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)

            # Calculate loss
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            val_losses.append(loss.item())
    mean_val_loss = mean(val_losses)
    val_losses.append(mean_val_loss)
    metrics['valid_loss'].append(mean_val_loss)

    print(f'Validation loss: {mean(val_losses)}')


    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        best_model_state = model.state_dict()
        # Save the best model
        torch.save(best_model_state, os.path.join(exp_root, exp_id, 'best_model.pth'))

plt.figure()
plt.plot(metrics['train_loss'], label='Train loss')
plt.plot(metrics['valid_loss'], label='Valid loss')
plt.legend()
plt.savefig(os.path.join(exp_root, exp_id, 'losses.png'))
