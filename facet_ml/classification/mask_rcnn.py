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

LABEL_TO_INT = {"B": 0, "C": 1, "MC": 2, "I": 3, "PS": 4, "V": 1}
INT_TO_LABEL = {0: "B", 1: "C", 2: "MC", 3: "I", 4: "PS"}


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
        self.n_images = len(self.filenames)

        # Load the h5s, then load the masks associated
        if not isinstance(h5_total, list):
            h5_files = [h5_total]
        else:
            h5_files = h5_total

        self.h5_files = h5_files

    def __getitem__(self, idx):

        # For the row of data, get its label and file name of interest
        fn_oi = self.filenames[idx]
        sub_df = self.df[self.df.Filename.apply(lambda x: str(Path(x).stem)) == fn_oi]
        h5_name = fn_oi
        # region_id = row_oi.Region
        # label = row_oi.Labels

        # From the h5s, grab the h5 associated with the image name
        h5_file = None
        for h5 in self.h5_files:
            if h5_name in h5.keys():
                h5_file = h5
                break
        if h5_file is None:
            raise Exception(f"'{h5_name}' does not exist in provided h5 files")

        # Grab the h5 markers and input image
        data_group = h5_file[h5_name]
        img = data_group["image_cropped"][:]
        if "markers_filled" in data_group.keys():
            markers_filled = data_group["markers_filled"][:]
        else:
            markers_filled = data_group["markers2"][:]

        obj_ids = np.unique(
            markers_filled
        )  # - 19  # Magic Number from increment in ImageSegmenter
        obj_ids = obj_ids[2:]
        num_objs = len(obj_ids)
        # Make masks for each class, but not including the edges (-20) and background (20)
        masks = torch.tensor(markers_filled == obj_ids[:, None, None]).to(
            dtype=torch.uint8
        )
        boxes = masks_to_boxes(
            masks,
        )

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Get labels from internal class dict
        sub_df.sort_values(by="Region", inplace=True)
        region_idxs = torch.tensor(sub_df.index.to_numpy())
        labels = torch.tensor([LABEL_TO_INT.get(l, 4) for l in sub_df.Labels])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        img = tv_tensors.Image(img)

        ## Finalize each of the structures to get rid of 0 area systems
        area_logical = area >= 1

        boxes_final = boxes[
            area_logical,
            :,
        ]
        area_final = area[area_logical]
        masks_final = masks[area_logical, :, :]
        iscrowd_final = iscrowd[area_logical]
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes_final, format="XYXY", canvas_size=F.get_size(img)
        )
        target["masks"] = tv_tensors.Mask(masks_final)
        target["labels"] = labels
        target["image_id"] = idx
        target["area"] = area_final
        target["iscrowd"] = iscrowd_final

        # Discard any boxes whose area is 0
        # temp_df = pd.DataFrame(target)
        # temp_df.drop(temp_df["area"] <= 1, inplace=True)
        # target = temp_df.to_dict()

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # print("IMAGE SHAPE:", np.shape(img))
        return img, target

        raise NotImplemented

    def __len__(self):
        return self.n_images
