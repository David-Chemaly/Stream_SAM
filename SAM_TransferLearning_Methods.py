import numpy as np

import torch
from torch.nn.functional import threshold, normalize
from segment_anything.utils.transforms import ResizeLongestSide

from collections import defaultdict
from segment_anything.utils.transforms import ResizeLongestSide

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

def preprocess_data(model, data, keys):
    '''
    data: array of images - shape (N, H, W, 1)
    keys: list of keys for the dictionary
    '''

    transformed_data = defaultdict(dict)
    for kndex, k in enumerate(keys):
        image = data[kndex]
        transform = ResizeLongestSide(model.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=DEVICE, dtype=DTYPE)
        transformed_image = input_image_torch.contiguous()[None, None, :, :]
        
        input_image = model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        transformed_data[k]['image'] = input_image
        transformed_data[k]['input_size'] = input_size
        transformed_data[k]['original_image_size'] = original_image_size

    return transformed_data
    
def forward(model,transformed_data, ground_truth_masks, bbox_coords, k, transform):
    '''
    transformed_data: dictionary of images from preprocess_data
    ground_truth_masks: dictionary of ground truth masks (labels)
    bbox_coords: dictionary of box coordinates
    k: key of the dictionary from the batch loop
    transform: transform to apply to the boxes from model
    '''

    input_image = transformed_data[k]['image'].to(DEVICE)
    input_size = transformed_data[k]['input_size']
    original_image_size = transformed_data[k]['original_image_size']

    # No grad here as we don't want to optimise the encoders
    with torch.no_grad():
        image_embedding = model.image_encoder(input_image)
    
        prompt_box = bbox_coords[k]
        box = transform.apply_boxes(prompt_box, original_image_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=DEVICE)
        box_torch = box_torch[None, :]
        
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

    low_res_masks, iou_predictions = model.mask_decoder(
    image_embeddings=image_embedding,
    image_pe=model.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=False,
    )

    upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).to(DEVICE)
    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

    gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[1], ground_truth_masks[k].shape[2]))).to(DEVICE)
    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

    return binary_mask, gt_binary_mask
