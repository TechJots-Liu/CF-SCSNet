import torch
import numpy as np
from scipy.ndimage import label
from PIL import Image

def process_binary_mask(tensor):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)
    
    binary_mask = tensor.squeeze(-1).cpu().numpy().astype(np.int8)

    labeled_array, num_features = label(binary_mask)
    
    filled_array = np.zeros_like(binary_mask)
    
    for i in range(1, num_features+1):
        region = (labeled_array == i)
        rows, cols = np.where(region)
        
        if len(rows) == 0:
            continue
            
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        filled_array[min_row:max_row+1, min_col:max_col+1] = 1

    return torch.from_numpy(filled_array).unsqueeze(-1).to(tensor.device)

def tensor_to_image(tensor):

    tensor = tensor.squeeze(-1).cpu().numpy() * 255
    tensor = tensor.astype(np.uint8)
    return Image.fromarray(tensor)