from facet_ml.segmentation import segmenter
from facet_ml.segmentation import visualization

from pathlib import Path
import json
import os
import numpy as np
import cv2
import copy

import torch


class CocoDataloader:
    """
    Purpose of this class is to load ground truth labeled data and its respective image
    for the purposes of comparing models
    """

    ID_INCREMENTER = 20

    def __init__(
        self,
        folder: str,
        annotations_file: str = "_annotations.coco.json",
        image_folder: str = None,
    ):
        '''
        Dataa class for loading Coco data, generally. Not used for ML purposes

        Args:
            folder (str) : Folder to look into
            annotations_file (str) : File that holds annotations for the images in the folder
            image_folder (str) : Folder to findd images in if different than the used folder 
        '''
        self.folder = folder
        self.annotations_file = annotations_file
        self.annotations_path = Path(folder) / annotations_file
        self.image_folder = image_folder or folder
        self._load_annotations()
        self.data = self._load_annotations()

    @property
    def n_images(self):
        return len(self.data["images"])

    @property
    def n_annotations(self):
        return len(self.data["images"])

    def _load_annotations(self):
        with open(str(self.annotations_path), "r") as f:
            data = json.load(f)

        return data

    def get_image_annotations(self, image_id):
        """
        Given an image, get its associated annotations
        """
        annotations = []
        for annotation in self.data["annotations"]:
            if annotation["image_id"] == image_id:
                annotations.append(annotation)
        return annotations

    def get_image_info(self, image_id):
        """
        Get information associated with an image
        """
        for image in self.data["images"]:
            if image["id"] == image_id:
                return image

    def get_image_path(self, image_id):
        """
        Get an image path (to later be used by OpenCV)
        """
        info = self.get_image_info(image_id)
        image_path = os.path.join(self.image_folder, info["file_name"])
        return image_path

    def get_categories(self):
        return {v["id"]: v["name"] for v in self.data["categories"]}

    def create_threshold(self, image_id):
        """
        For a given image id, create the background vs non-background threshold image

        This can be used for comparison of thresholding algorithms

        """
        annotations = self.get_image_annotations(image_id)
        categories = self.get_categories()
        info = self.get_image_info(image_id)
        thresh = np.zeros((info["height"], info["width"]), np.uint8)

        for annotation in annotations:
            label = categories[annotation["category_id"]]
            if label.lower() != "background":
                points = np.array(annotation["segmentation"]).reshape((-1, 2))
                cv2.fillPoly(thresh, [points.astype(np.int32)], 1)
        for annotation in annotations:
            if label.lower() == "background":
                points = np.array(annotation["segmentation"]).reshape((-1, 2))
                cv2.fillPoly(thresh, [points.astype(np.int32)], 0)

        return thresh

    def create_instance_segmentation(self, image_id):
        """
        For given image id, return an image mask where each instance is annotated as well as a dict for their labels

        """
        annotations = self.get_image_annotations(image_id)
        categories = self.get_categories()
        info = self.get_image_info(image_id)
        mask = np.zeros((info["height"], info["width"]), np.uint8)

        info_dict = {}
        priority_order = ["incomplete", "fused", "crystals"]
        for priority in priority_order:
            for ii, annotation in enumerate(annotations):
                label = categories[annotation["category_id"]]
                if label.lower() == priority:
                    points = np.array(annotation["segmentation"]).reshape((-1, 2))
                    cv2.fillPoly(
                        mask,
                        [points.astype(np.int32)],
                        ii + self.ID_INCREMENTER,
                    )
                    info_dict[annotation["id"]] = annotation
        for annotation in annotations:
            if label.lower() == "background":
                points = np.array(annotation["segmentation"]).reshape((-1, 2))
                cv2.fillPoly(
                    mask,
                    [points.astype(np.int32)],
                    0,
                )
        return mask, info_dict


def segment_mean_intersection_over_union(
    dataloader: CocoDataloader,
    image_segmenter: segmenter.ImageSegmenter,
    memoization: dict = {},
):
    """
    Given a CocoDataset and ImageSegmenter, compare against the ground truth for each image.
    Return thre mean iou AND
    """
    iou_vals = []
    for image_id in range(dataloader.n_images):
        image_path = dataloader.get_image_path(image_id)
        if image_path in memoization:
            image_segmenter = memoization[image_path]
        else:
            image_segmenter.input_path = image_path

        # Get necessary cropping information from the image segmenter
        tb = image_segmenter.top_boundary
        bb = image_segmenter.bottom_boundary
        lb = image_segmenter.left_boundary
        rb = image_segmenter.right_boundary

        ground_truth_mask = dataloader.create_threshold(image_id)
        ground_truth_mask = ground_truth_mask[tb:bb, lb:rb]
        ground_truth_mask = ground_truth_mask.astype(bool)

        pred_mask = image_segmenter.thresh.astype(bool)
        intersection = np.logical_and(ground_truth_mask, pred_mask).sum()
        union = np.logical_or(ground_truth_mask, pred_mask).sum()
        iou = intersection / union if union != 0 else 0.0
        iou_vals.append(iou)
        if image_path not in memoization:
            memoization[image_path] = copy.deepcopy(image_segmenter)
    return (np.mean(iou_vals), iou_vals, memoization)


def torch_segment_mean_intersection_over_union(
    dataloader: CocoDataloader, model, memoization: dict = {}, device="cuda"
):
    """
    Given a CocoDataset and Pytorch model, compare against the ground truth for each image.
    Return thre mean iou, respective vlaaues, and updated memoization
    Args:
        dataloader (CocoDataloader) : Class holding images to compare
        model (torch.model) : Torch model that can be run and used
        memoization (dict) : Hold memoized information for images such that rerunning is not required
        device (str) : Device used by torch
    """
    iou_vals = []
    for image_id in range(dataloader.n_images):
        image_path = dataloader.get_image_path(image_id)
        if image_path in memoization:
            out = memoization[image_path]
        else:
            model.to(device)
            model.eval()
            im = cv2.imread(image_path)
            im_torch = torch.tensor(np.moveaxis(im, -1, 0)) / 255
            im_torch = im_torch.to(device)
            im_torch = im_torch.unsqueeze(0)
            out = model(im_torch)[0]

        ground_truth_mask = dataloader.create_threshold(image_id)
        ground_truth_mask = ground_truth_mask.astype(bool)

        pred_masks_total = out["masks"]
        pred_masks_total = [
            mask.to("cpu").detach().numpy().astype(bool)
            for ii, mask in enumerate(pred_masks_total)
            if out["scores"][ii] >= 0.5
        ]
        pred_mask = np.sum(pred_masks_total, 0)
        intersection = np.logical_and(ground_truth_mask, pred_mask).sum()
        union = np.logical_or(ground_truth_mask, pred_mask).sum()
        iou = intersection / union if union != 0 else 0.0
        iou_vals.append(iou)
    return (np.mean(iou_vals), iou_vals, memoization)


def pixel_accuracy(
    dataloader: CocoDataloader,
    image_segmenter: segmenter.ImageSegmenter,
    memoization: dict = {},
):
    """
    Check how many pixels are correctly identified as background or not
    Args:
        dataloader (CocoDataloader) : Class holding images to compare
        model (torch.model) : Torch model that can be run and used
        memoization (dict) : Hold memoized information for images such that rerunning is not required
        
    """
    accuracy_vals = []
    for image_id in range(dataloader.n_images):
        image_path = dataloader.get_image_path(image_id)
        if image_path in memoization:
            image_segmenter = memoization[image_path]
        else:
            image_segmenter.input_path = image_path

        # Get necessary cropping information from the image segmenter
        tb = image_segmenter.top_boundary
        bb = image_segmenter.bottom_boundary
        lb = image_segmenter.left_boundary
        rb = image_segmenter.right_boundary

        ground_truth_mask = dataloader.create_threshold(image_id)
        ground_truth_mask = ground_truth_mask[tb:bb, lb:rb]
        ground_truth_mask = ground_truth_mask.astype(bool)

        pred_mask = image_segmenter.thresh.astype(bool)

        total_right = np.equal(ground_truth_mask, pred_mask).sum()
        total = len(pred_mask.ravel())
        accuracy = total_right / total
        accuracy_vals.append(accuracy)
        if image_path not in memoization:
            memoization[image_path] = copy.deepcopy(image_segmenter)
    return (np.mean(accuracy_vals), accuracy_vals, memoization)


def torch_pixel_accuracy(
    dataloader: CocoDataloader, model, memoization: dict = {}, device="cuda"
):
    """
    Check how many pixels are correctly identified as background or not
    Args:
        dataloader (CocoDataloader) : Class holding images to compare
        model (torch.model) : Torch model that can be run and used
        memoization (dict) : Hold memoized information for images such that rerunning is not required
        device (str) : Torch device to use
    """
    accuracy_vals = []
    for image_id in range(dataloader.n_images):
        image_path = dataloader.get_image_path(image_id)
        if image_path in memoization:
            out = memoization[image_path]
        else:
            model.to(device)
            model.eval()
            im = cv2.imread(image_path)
            im_torch = torch.tensor(np.moveaxis(im, -1, 0)) / 255
            im_torch = im_torch.to(device)
            im_torch = im_torch.unsqueeze(0)
            out = model(im_torch)[0]

        ground_truth_mask = dataloader.create_threshold(image_id)
        ground_truth_mask = ground_truth_mask.astype(bool)

        pred_masks_total = out["masks"]
        pred_masks_total = [
            mask.to("cpu").detach().numpy().astype(bool)
            for ii, mask in enumerate(pred_masks_total)
            if out["scores"][ii] >= 0.5
        ]
        pred_mask = np.sum(pred_masks_total, 0)

        total_right = np.equal(ground_truth_mask, pred_mask).sum()
        total = len(pred_mask.ravel())
        accuracy = total_right / total
        accuracy_vals.append(accuracy)

    return (np.mean(accuracy_vals), accuracy_vals, memoization)


def bidirectional_intersection(
    dataloader: CocoDataloader, image_segmenter: segmenter.ImageSegmenter
):
    """
    Get the bidirectional comparison matrix
    Args:
        dataloader (CocoDataloader) : Class holding images to compare
        image_segmenter (ImageSegmenter) : ImageSegmenter being compared
    """

    BD_arr = []
    d1_arr = []
    d2_arr = []
    for image_id in range(dataloader.n_images):
        image_path = dataloader.get_image_path(image_id)
        image_segmenter.input_path = image_path

        # Get necessary cropping information from the image segmenter
        tb = image_segmenter.top_boundary
        bb = image_segmenter.bottom_boundary
        lb = image_segmenter.left_boundary
        rb = image_segmenter.right_boundary

        annotations = dataloader.get_image_annotations(image_id)
        ground_truth_image, ground_truth_dict = dataloader.create_instance_segmentation(
            image_id
        )
        N = len(image_segmenter.region_dict)
        M = len(ground_truth_dict)

        d1 = np.zeros((N, M))
        d2 = np.zeros((N, M))

        # Match against each region
        regions_dict_is = image_segmenter.grab_region_dict(focused=False, alpha=0)
        for ii, n in enumerate(regions_dict_is.keys()):
            region_is = regions_dict_is[n]

            for jj, m in enumerate(range(M)):
                region_gt = ground_truth_image == (m + dataloader.ID_INCREMENTER)
                region_gt = region_gt[tb:bb, lb:rb]

                intersection = np.logical_and(region_is, region_gt).sum()
                mask1 = np.isin(region_is, region_gt)
                diff1 = region_is[mask1]
                d1[ii, jj] = intersection / region_is.sum()

                mask2 = np.isin(region_gt, region_is)
                diff2 = region_gt[mask2]
                d2[ii, jj] = intersection / region_gt.sum()

        d1_arr.append(d1)
        d2_arr.append(d2)
        BD_arr.append(2 / ((1 / d1) + (1 / d2)))

    # For each of these arrays
    return BD_arr, d1_arr, d2_arr


def instance_mean_intersection_over_union(
    dataloader: CocoDataloader,
    image_segmenter: segmenter.ImageSegmenter,
    memoization: dict = {},
):
    """
    Get the mIoU
    Args:
        dataloader (CocoDataloader) : Class holding images to compare
        model (torch.model) : Torch model that can be run and used
        memoization (dict) : Hold memoized information for images such that rerunning is not required
        
    """

    iou_total_values = np.array([])
    for image_id in range(dataloader.n_images):
        image_path = dataloader.get_image_path(image_id)
        if image_path in memoization:
            image_segmenter = memoization[image_path]
        else:
            image_segmenter.input_path = image_path

        # Get necessary cropping information from the image segmenter
        tb = image_segmenter.top_boundary
        bb = image_segmenter.bottom_boundary
        lb = image_segmenter.left_boundary
        rb = image_segmenter.right_boundary

        annotations = dataloader.get_image_annotations(image_id)
        ground_truth_image, ground_truth_dict = dataloader.create_instance_segmentation(
            image_id
        )

        regions_dict_is = image_segmenter.grab_region_dict(focused=False, alpha=0)

        N = len(regions_dict_is)
        M = len(ground_truth_dict)

        iou_arr = np.zeros((N, M))
        for ii, n in enumerate(regions_dict_is.keys()):
            region_is = regions_dict_is[n].astype(bool)

            for jj, m in enumerate(range(M)):
                region_gt = ground_truth_image == (m + dataloader.ID_INCREMENTER)
                region_gt = region_gt[tb:bb, lb:rb].astype(bool)

                intersection = np.logical_and(region_is, region_gt).sum()
                union = np.logical_or(region_is, region_gt).sum()

                iou = intersection / union
                try:
                    iou_arr[ii, jj] = iou
                except:
                    Exception(f"{regions_dict_is},{N},{M}")

        if (N > 0) and (M > 0):
            iou_values = np.max(iou_arr, axis=0)
            iou_total_values = np.concatenate([iou_total_values, iou_values])
        if image_path not in memoization:
            memoization[image_path] = copy.deepcopy(image_segmenter)

    iou_nonzeros = iou_total_values[iou_total_values != 0]
    return (
        np.mean(iou_total_values),
        np.mean(iou_nonzeros),
        iou_total_values,
        memoization,
    )


def torch_instance_mean_intersection_over_union(
    dataloader: CocoDataloader, model, memoization: dict = {}, device="cuda"
):
    """
    Get the mIoU
    Args:
        dataloader (CocoDataloader) : Class holding images to compare
        model (torch.model) : Torch model that can be run and used
        memoization (dict) : Hold memoized information for images such that rerunning is not required
        device (str) : Torch device to use
    """

    iou_total_values = np.array([])
    for image_id in range(dataloader.n_images):
        image_path = dataloader.get_image_path(image_id)
        if image_path in memoization:
            out = memoization[image_path]
        else:
            model.to(device)
            model.eval()
            im = cv2.imread(image_path)
            im_torch = torch.tensor(np.moveaxis(im, -1, 0)) / 255
            im_torch = im_torch.to(device)
            im_torch = im_torch.unsqueeze(0)
            out = model(im_torch)[0]

        pred_masks_total = out["masks"]
        pred_masks_total = [
            mask for ii, mask in enumerate(pred_masks_total) if out["scores"][ii] >= 0.5
        ]

        annotations = dataloader.get_image_annotations(image_id)
        ground_truth_image, ground_truth_dict = dataloader.create_instance_segmentation(
            image_id
        )

        pred_masks_total = [
            mask.to("cpu").detach().numpy().astype(bool) for mask in pred_masks_total
        ]

        N = len(pred_masks_total)
        M = len(ground_truth_dict)

        iou_arr = np.zeros((N, M))
        for ii, region in enumerate(pred_masks_total):
            region = region.astype(bool)

            for jj, m in enumerate(range(M)):
                region_gt = ground_truth_image == (m + dataloader.ID_INCREMENTER)

                intersection = np.logical_and(region, region_gt).sum()
                union = np.logical_or(region, region_gt).sum()

                iou = intersection / union
                try:
                    iou_arr[ii, jj] = iou
                except:
                    Exception(f"{pred_masks_total},{N},{M}")

        if (N > 0) and (M > 0):
            iou_values = np.max(iou_arr, axis=0)
            iou_total_values = np.concatenate([iou_total_values, iou_values])

    iou_nonzeros = iou_total_values[iou_total_values != 0]
    return (
        np.mean(iou_total_values),
        np.mean(iou_nonzeros),
        iou_total_values,
        memoization,
    )
