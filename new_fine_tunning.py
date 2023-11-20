import h5py
import monai
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import  datasets

from transformers import SamProcessor
from transformers import SamModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

# Parameters
lr = 1e-5
wd = 0
num_epochs = 100
bs = 1

path_input  = '/Volumes/ES-HDD-Documents/Documents/CFHT_galaxies_with_streams'
path_output = '.'

# Load Train and Valid data
n = 5
with h5py.File(f'{path_input}/train_data.h5', 'r') as f:
    train_images = (f['images'][:n,0,:,:]*255).astype(np.uint8) # (N, H, W)
    train_masks  = f['masks'][:n,0,:,:]  # (N, H, W)
H = train_images.shape[1]
W = train_images.shape[2]

with h5py.File(f'{path_input}/val_data.h5', 'r') as f:
    valid_images = (f['images'][:n,0,:,:]*255).astype(np.uint8) # (N, H, W)
    valid_masks  = f['masks'][:n,0,:,:]  # (N, H, W)

# Convert the NumPy arrays to Pillow images and store them in a dictionary
train_dataset_dict = {
    "image": [Image.fromarray(img) for img in train_images],
    "label": [Image.fromarray(mask) for mask in train_masks],
}

valid_dataset_dict = {
    "image": [Image.fromarray(img) for img in valid_images],
    "label": [Image.fromarray(mask) for mask in valid_masks],
}

# Create the dataset using the datasets.Dataset class
train_dataset = datasets.Dataset.from_dict(train_dataset_dict)
valid_dataset = datasets.Dataset.from_dict(valid_dataset_dict)

#Get bounding boxes from mask.
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

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs
  
# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the SAMDataset
train_dataset = SAMDataset(dataset=train_dataset, processor=processor)
valid_dataset = SAMDataset(dataset=valid_dataset, processor=processor)

# Create a DataLoader instance for the training dataset
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, drop_last=False)

# Load the model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

#Training and Validation loop
model.to(DEVICE)

loss_train_epoch = []
loss_valid_epoch = []
for epoch in tqdm(range(num_epochs), leave=True):

    loss_train_batch = []
    loss_valid_batch = []   

    # Training 
    model.train()
    for batch in train_dataloader:
      
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(DEVICE),
                      input_boxes=batch["input_boxes"].to(DEVICE),
                      multimask_output=False)

      # compute loss
      ground_truth_masks = batch["ground_truth_mask"].float().to(DEVICE)
      predicted_masks = outputs.pred_masks.squeeze(1)
      upscaled_predicted_masks = torch.nn.functional.interpolate(predicted_masks, size=(ground_truth_masks.shape[-2],ground_truth_masks.shape[-1]))
      loss = seg_loss(upscaled_predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      loss_train_batch.append(loss.item())

    # Validation
    model.eval()
    for batch in valid_dataloader:
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(DEVICE),
                      input_boxes=batch["input_boxes"].to(DEVICE),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      upscaled_predicted_masks = torch.nn.functional.interpolate(predicted_masks, size=(H,W))
      ground_truth_masks = batch["ground_truth_mask"].float().to(DEVICE)
      loss = seg_loss(upscaled_predicted_masks, ground_truth_masks.unsqueeze(1))
      
      loss_valid_batch.append(loss.item())

    loss_train_epoch.append(np.mean(loss_train_batch))
    loss_valid_epoch.append(np.mean(loss_valid_batch))

    print(f'EPOCH: {epoch}')
    print(f'Train loss: {loss_train_epoch[epoch]}, Valid loss: {loss_valid_epoch[epoch]}')

    # Save and Plot loss
    np.savetxt(f'{path_output}/train_losses.txt', loss_train_epoch)
    np.savetxt(f'{path_output}/valid_losses.txt', loss_valid_epoch)
    plt.plot(loss_train_epoch, color='blue', label='train')
    plt.plot(loss_valid_epoch, color='orange', label='valid')
    plt.semilogy()
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{path_output}/loss.png')
    plt.close()

    # Save model
    torch.save(model.state_dict(), f'{path_output}/finetuned_SAM.pth')