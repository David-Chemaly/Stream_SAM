import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import h5py

import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from SAM_TransferLearning_Methods import preprocess_data, forward

DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

# Load index
data_path    = "/Volumes/ES-HDD-Documents/Documents/CFHT_galaxies_with_streams/"
label_path   = "/Volumes/ES-HDD-Documents/Documents/CFHT_galaxies_with_streams/streams_masks/"
data_list    = np.loadtxt(data_path + "list_asinh.txt", dtype=str)
label_list   = np.loadtxt(label_path + "/list_mask.txt", dtype=str)
galaxy_names = np.loadtxt(data_path + "list_galaxy_names.txt", dtype=str)

# Load data
n = 2
with h5py.File(data_path+'train_data.h5', 'r') as f:
    train_images = f['images'][:n] # (N, 1, H, W)
    train_masks  = f['masks'][:n]  # (N, 1, H, W)
train_images = np.transpose((train_images*255).astype(np.uint8), (0, 2, 3, 1)) # From NCHW to NHWC

with h5py.File(data_path+'val_data.h5', 'r') as f:
    valid_images = f['images'][:n] # (N, 1, H, W)
    valid_masks  = f['masks'][:n]  # (N, 1, H, W)
valid_images = np.transpose((valid_images*255).astype(np.uint8), (0, 2, 3, 1)) # From NCHW to NHWC


# Load boxes and masks in Dictionary for Training Set
train_bbox_coords = {}
valid_bbox_coords = {}
train_ground_truth_masks = {}
valid_ground_truth_masks = {}
for i in range(train_images.shape[0]):
    train_ground_truth_masks[i] = train_masks[i].astype(bool)
    train_bbox_coords[i] = np.array([0,0,train_images.shape[-2],train_images.shape[-1]])

    # Load boxes and masks in Dictionary for Validation Set
    if i < valid_images.shape[0]:
        valid_ground_truth_masks[i] = valid_masks[i].astype(bool)
        valid_bbox_coords[i] = np.array([0,0,valid_images.shape[-2],valid_images.shape[-1]])

# Load model
sam_checkpoint = "/Users/davidchemaly/Weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_model.to(device=DEVICE)


transform = ResizeLongestSide(sam_model.image_encoder.img_size)

train_keys = list(train_bbox_coords.keys())
valid_keys = list(valid_bbox_coords.keys())
train_transformed_data = preprocess_data(sam_model.train(), train_images, train_keys)
valid_transformed_data = preprocess_data(sam_model.eval(), valid_images, valid_keys)

# Set up the optimizer, hyperparameter tuning will improve performance here
lr = 1e-4
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCEWithLogitsLoss()

# Train the model
num_epochs = 100
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
  train_epoch_losses = []
  valid_epoch_losses = []

  # Shuffle data
  np.random.shuffle(train_keys)
  np.random.shuffle(valid_keys)

  for tk in tqdm(train_keys, leave=True):

    train_binary_mask, train_gt_binary_mask = forward(sam_model.train(), train_transformed_data, train_ground_truth_masks, train_bbox_coords, tk, transform)

    train_loss = loss_fn(train_binary_mask, train_gt_binary_mask)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    train_epoch_losses.append(train_loss.item())

  for vk in tqdm(valid_keys, leave=True):
     
    valid_binary_mask, valid_gt_binary_mask = forward(sam_model.eval(), valid_transformed_data, valid_ground_truth_masks, valid_bbox_coords, tk, transform)
    
    valid_loss = loss_fn(valid_binary_mask, valid_gt_binary_mask)
    valid_epoch_losses.append(valid_loss.item())


  train_losses.append(np.mean(train_epoch_losses))
  valid_losses.append(np.mean(valid_epoch_losses))
  print(f'EPOCH: {epoch}')
  print(f'TRAIN loss: {train_losses[-1]} & VALID loss: {valid_losses[-1]}')

# Save the model
torch.save(sam_model.state_dict(), "/Users/davidchemaly/Weights/sam_vit_h_4b8939_fine_tuned.pth")

# Plot the losses
np.save('train_losses.txt', train_losses)
np.save('valid_losses.txt', valid_losses)
plt.plot(train_losses, color='blue', label='train')
plt.plot(valid_losses, color='orange', label='valid')
plt.semilogy()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()