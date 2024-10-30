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
import cv2
import json
import os
from pathlib import Path

from pycocotools.coco import COCO



### NOTE: These apply only for the manual h5 labeling
# Labels for variable conversion and haandling some common typos
LABEL_TO_INT = {"B": 0, "C": 1, "MC": 2, "I": 3, "PS": 4, "V": 1}
INT_TO_LABEL = {0: "B", 1: "C", 2: "MC", 3: "I", 4: "PS"}


def validate_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 > x1) and (y2 > y1)


class ColloidalDataset(Dataset):

    def __init__(self, df_total, h5_total: list, transforms=None):
        """
        Given the dataframe of each file,
        associate the data rows to their binary masks in the h5s made via ImageSegmenter.to_h5
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
            markers_filled = data_group["markers2"][:] # Old naming convention

        obj_ids = np.unique(
            markers_filled
        ) 
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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.n_images
    

class ManualCocoColloidalDataset(Dataset):
    """
    This dataset is intended to be used w/ Coco labeled data
    """

    def __init__(self, root, annotation_file, transforms=None):
        """
        Args:
            root (string): Root directory where images are stored.
            annotation_file (string): Path to the COCO annotations file.
            transforms (callable, optional): A function/transform that takes in
                                             an image and returns a transformed version.
        """
        self.root = root
        self.transforms = transforms
        self.annotation_file = annotation_file

        # Load annotations
        with open(annotation_file, "r") as f:
            self.coco = json.load(f)

        # Get all images and annotations
        self.images = self.coco["images"]
        self.annotations = {ann["image_id"]: [] for ann in self.coco["annotations"]}
        for ann in self.coco["annotations"]:
            self.annotations[ann["image_id"]].append(ann)

        # Get category ID to class label
        self.category = {cat["id"]:cat["name"] for cat in self.coco["categories"]}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image and annotations
        img_info = self.images[idx]
        img_id = int(img_info["id"])
        img_path = os.path.join(self.root, img_info["file_name"])

        # Load image with cv2
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        # Get the annotations for the current image
        annotations = self.annotations[img_id]

        boxes = []
        labels = []
        masks = []
        image_ids = []
        for ann in annotations:
            bbox = ann["bbox"]
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]

            # Some images, labeled background explicitly. Handle this
            if "background" in self.category[ann["category_id"]].lower():
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(not "background" in self.category[ann["category_id"]].lower())
            image_ids.append(ann["image_id"])

            empty = np.zeros(image.shape[:2])

            points = np.array(ann["segmentation"]).reshape((-1, 2))
            cv2.fillPoly(empty, [points.astype(np.int32)], 1)
            masks.append(empty)

        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.int64)
        masks = tv_tensors.Mask(masks)
        # boxes = masks_to_boxes(torch.tensor(masks))

        # Tensor-fy it
        image = tv_tensors.Image(np.moveaxis(image, -1, 0)) / 255

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(image)
        )
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["masks"] = tv_tensors.Mask(masks)

        # Apply transforms
        if self.transforms:
            image, target["masks"] = self.transforms(image, target["masks"])

        # Develop into final representations
        if not target["masks"].any():
            # Handle case where all masks become empty after transforms
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            new_labels = []
            new_masks = []
            new_boxes = []
            for ii, mask in enumerate(target["masks"]):
                if not mask.any():
                    continue
                box = masks_to_boxes(mask.unsqueeze(0))[0]
                if not validate_bbox(box):
                    continue
                new_boxes.append(box)
                new_labels.append(target["labels"][ii])
                new_masks.append(mask)

            target["masks"] = torch.stack(new_masks)
            boxes = torch.stack(new_boxes)

        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(image)
        )

        return image, target


class CocoColloidalDataset(Dataset):
    """
    This dataset is intended to be used w/ Coco labeled data using pycocotools
    NOTE: Encountered issues with this form, swapping away but leaving here 
        for future editing
    """

    def __init__(self, root, annotation_file, transforms=None):
        """
        Args:
            root (string): Root directory where images are stored.
            annotation_file (string): Path to the COCO annotations file.
            transforms (callable, optional): A function/transform that takes in
                                             an image and returns a transformed version.
        """
        self.root = root
        self.transforms = transforms
        self.annotation_file = annotation_file
        self.coco = COCO(self.annotation_file)
        self.ids = list(self.coco.imgs.keys())

        # # Load annotations
        # with open(annotation_file, "r") as f:
        #     self.coco = json.load(f)

        # # Get all images and annotations
        # self.images = self.coco["images"]
        # self.annotations = {ann["image_id"]: [] for ann in self.coco["annotations"]}
        # for ann in self.coco["annotations"]:
        #     self.annotations[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get the image and annotations
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.root, img_info["file_name"])

        # Load image with cv2
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        # Get the annotations for the current image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        num_objs = len(anns)
        boxes = []
        labels = []
        masks = []

        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])
            masks.append(self.coco.annToMask(ann))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack(
            [torch.as_tensor(mask, dtype=torch.uint8) for mask in masks]
        )

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
