from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import h5py
import numpy as np
import pandas as pd
import glob

from pathlib import Path

LABEL_TO_INT = {"B": 0, "C": 1, "MC": 2, "I": 3, "PS": 4, "V": 1, "PC":4}
INT_TO_LABEL = {0: "B", 1: "C", 2: "MC", 3: "I", 4: "PS"}

def crop_to_nonzero(image,padding=5):
    """
    Crop a NumPy array image to the bounding box of its non-zero regions.
    
    Args:
        image (np.array): The input image array.
        
    Returns:
        np.array: The cropped image array.
    """
    # Find the non-zero elements in the array
    non_zero_indices = np.nonzero(image)
    
    # Get the bounding box coordinates
    min_row, min_col = np.min(non_zero_indices[0]), np.min(non_zero_indices[1])
    max_row, max_col = np.max(non_zero_indices[0]), np.max(non_zero_indices[1])
    
    # Crop the image to the bounding box
    cropped_image = image[min_row:max_row+1, min_col:max_col+1]

    # Pad image
    pad_image = np.pad(cropped_image,padding,mode="constant",constant_values=0)
    
    return pad_image



class ColloidalDataset(Dataset):

    def __init__(self, df_total, h5_total: list, transforms=None):
        """
        Given the dataframe of each file,
        associate the data rows to their binary masks in the h5s.
        """
        self.transforms = transforms

        # Load the row dataframe, organize it by filename
        self.df: pd.DataFrame = df_total
        self.df.reset_index(drop=True, inplace=True)
        self.df.sort_values(by="Filename", inplace=True)
        self.filenames = [Path(fn).stem for fn in self.df.Filename.unique()]
        self.n_images = len(self.df)

        # Load the h5s, then load the masks associated
        if not isinstance(h5_total, list):
            h5_files = [h5_total]
        else:
            h5_files = h5_total

        self.h5_files = h5_files

    def __getitem__(self, idx):

        # Get Row of data
        row = self.df.iloc[idx]

        # For row of the data get its label, file name, and region
        label = row.Labels
        filename = str(Path(row.Filename).stem)
        region = row.Region

        # From the h5s, grab the h5 associated with the image name
        h5_name = filename
        h5_file = None
        for h5 in self.h5_files:
            if h5_name in h5.keys():
                h5_file = h5
                break
        if h5_file is None:
            raise Exception(f"'{h5_name}' does not exist in provided h5 files")

        # Grab the h5 regions
        data_group = h5_file[h5_name]
        region_index = region - 1 

        img = data_group["Regions"][region_index,:,:]

        # Apply bounding
        img = crop_to_nonzero(img)

        # target = temp_df.to_dict()

        if self.transforms is not None:
            img = self.transforms(img)

        # a = np.zeros( (len(INT_TO_LABEL),1) )
        # a[LABEL_TO_INT[label]] = 1
        # return img, torch.tensor(a)
        return img, LABEL_TO_INT.get(label,4)

        raise NotImplemented

    def __len__(self):
        return self.n_images
