# This is an improvement to the segmentation_utils class to make it more general

import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from skimage import measure, color, io
import math
import copy
import pandas as pd

import torch

pd.options.mode.chained_assignment = None  # default='warn'
import os
from IPython.display import clear_output
from functools import partial
from skimage import data, segmentation, feature, future
from skimage.filters import threshold_local
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path
from typing import Union
import h5py

from facet_ml.segmentation import edge_modification as em
from facet_ml.segmentation import thresholding
from facet_ml.static.path import STATIC_MODELS
from facet_ml.segmentation import features as feat


from abc import ABC, abstractmethod, abstractproperty


# Legacy incrementer for image pixels
LABEL_INCREMENT = 20

#### Segmenters ####
# These just act on a given image, no foo-foo processing
class AbstractSegmenter(ABC):

    @abstractmethod
    def __init__(self, image):
        """
        Abstractable Class for creating segmentation schemes

        Args:
            image (np.ndarray) : Image to be segmented
        """
        self._image = image

        self._label_increment = LABEL_INCREMENT
        # Class variables
        self._thresh = None
        self._image_working = None
        self._image_labeled = None
        self._markers = None  # Seeds for regions
        self._markers_filled = None  # The actual regions of interest

    def reset_segmenter(self):
        """
        For a segmenter, remove all internal images
        Readies for re-use on new image
        """
        self.image
        self._thresh = None
        self._image_working = None
        self._image_labeled = None
        self._markers = None  # Seeds for regions
        self._markers_filled = None  # The actual regions of interest

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, new_image):
        self._image = new_image
        self.reset_segmenter()

    @abstractproperty
    def thresh(self):
        raise NotImplemented

    @property
    def image_working(self):
        raise NotImplemented

    @property
    def image_labeled(self):
        raise NotImplemented

    @property
    def markers(self):
        raise NotImplemented

    @abstractproperty
    def markers_filled(self):
        raise NotImplemented


class AlgorithmicSegmenter(AbstractSegmenter):

    # Add new connections to thresholding functions as developed
    mapping_thresh = [
        (lambda tm: tm == "otsu", thresholding.otsu_threshold),
        (lambda tm: tm == "local", thresholding.local_threshold),
        (lambda tm: tm == "pixel", thresholding.pixel_threshold),
        (lambda tm: tm == "ensemble", thresholding.ensemble_threshold),
        (lambda tm: isinstance(tm, list), thresholding.multi_threshold),
    ]

    # Add new connections to edge thresholding functions as developed
    mapping_edge = [
        (
            lambda edge: edge == None,
            lambda s: np.full_like(s.image, 0).astype(np.uint8),
        ),
        (lambda edge: edge == "canny", em.edge_canny),
        (lambda edge: edge == "variance", em.edge_variance),
        (lambda edge: edge == "darkbright", em.edge_darkbright),
        (lambda edge: edge == "classifier", em.edge_classifier),
        (lambda edge: edge == "localthresh", em.edge_localthresh),
        (lambda edge: edge == "testing", em.edge_testing),
    ]

    def __init__(
        self,
        image: np.ndarray,
        threshold_mode="otsu",
        edge_modification: str = None,
        kernel: np.ndarray = np.ones((3, 3), np.uint8),
    ):
        """
        Create a Segmenter class which has mappings to
        mapping_thresh and mapping_edge

        Args:
            image (np.ndarray) : Image to be segmented
            threshold_mode (str | list) : Use a method that returns a threshold.
                    Provide a list for ensemble or custom algorithms
            edge_modification (str) : Additional algorithm for adding edges to threshold methods
            kernel (np.ndarray) : Kernel used for morphological transforms (erode, dilate)
        """

        super().__init__(image)

        # Implementation details
        self.threshold_mode = threshold_mode
        self.edge_modification = edge_modification
        self.kernel = kernel

        # Model details
        self.pixel_model = None

    @property
    def thresh(self):
        if self._thresh is None:

            # Get the basic threshold
            # mapping : (bool fun, fun_to_call)

            thresh = None
            success = False
            for bool_fun, mapped_fun in AlgorithmicSegmenter.mapping_thresh:
                if bool_fun(self.threshold_mode):
                    thresh = mapped_fun(self)
                    success = True
                    break
            if not success:
                raise Exception(f"{self.threshold_mode} not supported.")

            # Morphologically manipulate threshold
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel, iterations=2)
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_CLOSE, self.kernel, iterations=1
            )

            # Apply edge considerations
            image_edges = None
            for bool_fun, mapped_fun in AlgorithmicSegmenter.mapping_edge:
                if bool_fun(self.edge_modification):
                    image_edges = mapped_fun(self)
            if image_edges is None:
                raise Exception(f"{self.edge_modification} not supported")

            self._thresh = thresh - image_edges

        return self._thresh

    @property
    def markers(self):
        if self._markers is None:
            # Note: May need to do this before the edge_modification
            self._bg_mark = cv2.dilate(self.thresh, self.kernel, iterations=1)
            thresh_border = cv2.copyMakeBorder(
                self.thresh,
                top=1,
                bottom=1,
                right=1,
                left=1,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )
            self._dist_transform = cv2.distanceTransform(thresh_border, cv2.DIST_L2, 5)
            self._dist_transform = self._dist_transform[1:-1, 1:-1]

            # Sure foreground
            scaling_rule = self._dist_transform.max() * 0.35
            ret2, fg_mark = cv2.threshold(self._dist_transform, scaling_rule, 255, 0)
            fg_mark = cv2.erode(fg_mark, self.kernel)
            self._fg_mark = np.uint8(fg_mark)

            ## Develop unknown region
            self.unknown = cv2.subtract(self._bg_mark, self._fg_mark)

            # Develop Regions
            self.outputs = cv2.connectedComponentsWithStats(self._fg_mark)
            self._markers = self.outputs[1] + self._label_increment
            self._markers[self.unknown == 255] = 0
        return self._markers

    @property
    def markers_filled(self):
        if self._markers_filled is None:
            temp_markers = copy.deepcopy(self.markers)

            image_blur = cv2.cvtColor(
                cv2.GaussianBlur(self.image, (9, 9), 0), cv2.COLOR_GRAY2RGB
            )
            self._markers_filled = cv2.watershed(image_blur, temp_markers)
        return self._markers_filled

    def load_pixel_segmenter(self):
        """
        Load the pixel classifer. This is a LARGE model, so only use this if needed
        """
        if not self.pixel_model:
            with open(STATIC_MODELS["bg_segmenter"], "rb") as f:
                self.pixel_model = pickle.load(f)

class MaskRCNNSegmenter(AbstractSegmenter):

    def __init__(self, image: np.ndarray, device: str = None):
        """
        Create a Segmenter class which uses a MaskRCNN model from detectron2

        Args:
            image (np.ndarray) : Image to be segmented
            device (str) : Device information for Torch
        """
        super().__init__(image)

        import torch

        folder_path = os.path.join(
            Path(__file__).parent.parent, "static", "Models", "torch"
        )
        model_path = STATIC_MODELS["maskrcnn"]
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        elif torch.cuda.is_available() and "cuda" in device:
            self.device = device
        else:
            self.device = "cpu"

        self.model = torch.load(model_path)
        self.model.to(self.device)

    @property
    def thresh(self):
        if self._thresh is None:
            self._thresh = self.markers_filled > 0
        return self._thresh

    @property
    def markers(self):
        if self._markers is None:
            self._markers = np.full_like(self.image, 0)
        return self._markers

    @property
    def markers_filled(self):
        import torch

        if self._markers_filled is None:
            # Load the image into the model
            mod_img = (
                self.image
                if len(self.image) == 3
                else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            )
            mod_img_torch = (
                torch.tensor(np.moveaxis(mod_img, -1, 0)).to(self.device) / 255
            )

            out = self.model([mod_img_torch])[0]
            self.out = out
            masks = out["masks"].to("cpu").detach().numpy()
            scores = out["scores"].to("cpu").detach().numpy()
            mask_stack = [mask for ii, mask in enumerate(masks) if scores[ii] >= 0.5]
            if len(mask_stack) == 0:
                mask_stack = np.zeros((1, 1, *np.shape(self.image)))
            masks = np.stack(mask_stack)

            # Note to self: skimage.measure.label to leverage detectron2 model as a labeler
            self._markers_filled = self._label_increment * np.ones(np.shape(self.image))
            num_markers, _, _, _ = np.shape(masks)
            for ii in np.arange(num_markers):

                mask_oi = masks[ii, 0, :, :]
                mask_oi = np.where(mask_oi > .5,True,False)
                mask_bulk = cv2.erode(
                    mask_oi.astype(np.uint8), kernel=np.ones((3, 3))
                ).astype(bool)
                mask_edge = ~mask_bulk & mask_oi
                self._markers_filled[mask_edge] = -1
                self._markers_filled[mask_bulk] = 1 + self._label_increment + ii
            self._markers_filled = self._markers_filled.astype(int)
        return self._markers_filled


class SAMSegmenter(AbstractSegmenter):
    def __init__(
        self,
        image: np.ndarray,
        device: str = None,
        sam_kwargs: dict = {
            "points_per_side": 64, 
            "min_mask_region_area":20
                            },
    ):
        """
        Create a Segmenter class which uses a SegmentAnything model from Meta.
        This uses the AutomateddMaskGenerator by gridpoints

        Args:
            image (np.ndarray) : Image to be segmented
            device (str) : specify if using cuda or cpu for Torch
            sam_kwargs (dict) : Kwargs for segment_anything.SamAutomaticMaskGenerator
                            of segment anything
        """

        super().__init__(image)
        self.sam_kwargs = sam_kwargs
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        elif torch.cuda.is_available() and "cuda" in device:
            self.device = device
        else:
            self.device = "cpu"

        # SAM Variable
        self._mask_generator = None

    @property
    def mask_generator(self):
        if self._mask_generator is None:

            from segment_anything import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
            
            model_type = "vit_l"
            sam_checkpoint = STATIC_MODELS["segment_anything_vit_l"]
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            mask_generator = SamAutomaticMaskGenerator(sam, **self.sam_kwargs)
            self._mask_generator = mask_generator
        return self._mask_generator

    @property
    def thresh(self):
        """
        NOT a determiner for other work on this segmenter
        Can set it just to be where there ar ergions
        """
        if self._thresh is None:
            self._thresh = self.markers_filled > 0
        return self._thresh

    @property
    def markers_filled(self):
        if self._markers_filled is None:
            image_convert = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
            with torch.no_grad():
                with torch.device(self.device):
                    masks = self.mask_generator.generate(image_convert)

                self._markers_filled = self._label_increment * np.ones(np.shape(self.image))
                for ii, mask in enumerate(masks):
                    mask_oi = mask["segmentation"]
                    mask_bulk = cv2.erode(
                        mask_oi.astype(np.uint8), kernel=np.ones((3, 3))
                    ).astype(bool)
                    mask_edge = ~mask_bulk & mask_oi
                    self._markers_filled[mask_edge] = -1
                    self._markers_filled[mask_bulk] = 1 + self._label_increment + ii
                self._markers_filled = self._markers_filled.astype(int)
        return self._markers_filled

# Add to this mapper as new segmenters are added
segmenter_mapper = {
    "maskrcnn": MaskRCNNSegmenter,
    "segment_anything": SAMSegmenter,
    "algorithmic": AlgorithmicSegmenter,
}


class ImageSegmenter:

    def __init__(
        self,
        input_path: str = None,
        pixels_to_um: float = 9.37,
        top_boundary: int = 0,
        bottom_boundary: int = 860,
        left_boundary: int = 0,
        right_boundary: int = 2560,
        result_folder_path: str = "Results",
        override_exists: bool = False,
        filename: str = None,
        segmenter: str = "algorithmic",
        segmenter_kwargs: dict = {},
        region_featurizers: list = [
            feat.AverageCurvatureFeaturizer(),
            feat.StdCurvatureFeaturizer(),
            feat.MinCurvatureFeaturizer(),
            feat.MaxCurvatureFeaturizer(),
            feat.PercentConvexityCurvatureFeaturizer(),
            feat.LongestContiguousConcavityCurvatureFeaturizer(),
            feat.LongestContiguousConvexityCurvatureFeaturizer(),
            feat.DistinctPathsCurvatureFeaturizer(),
        ],
        file_str: str = None,
    ):
        """
        Main class for handling segmentation pipeline. Encapsulates reading an image (or image path), loading a segmenter, applying features, and 
        assisted labeling.

        Args:
            input_path (string OR img)    : Path to the image desired to be interpreted (if img, create tmp file)
            pixels_to_um (float)   : Scale factor for image analysis and featurization
            top_boundary (int)     : Pixel boundary for cropping image
            bottom_boundary (int)  : Pixel boundary for cropping image
            left_boundary (int)    : Pixel boundary for cropping image
            right_boundary (int)   : Pixel boundary for cropping image
            result_folder_path (string) : Path to the folder .csv should be saved.
            override_exists (bool) : If .csv already exists, DO NOT overwrite it if this variable is False. Allows classification across sessions
        """
        # Given variables (besides input path, handle that at very end)
        self._input_path = input_path
        self.pixels_to_um = pixels_to_um
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.result_folder_path = result_folder_path
        self.override_exists = override_exists
        self.file_str = file_str
        self.segmenter_kwargs = segmenter_kwargs
        self.region_featurizers = region_featurizers

        # Image variables
        self._image_read = None
        self._image_cropped = None
        self._image_working = None
        self._image_labeled = None
        self._thresh = None

        # Define default image variables
        # NOTE: Will need to do something with this in the future for input
        self.canny_tl = 40
        self.canny_tu = 40
        self.blur_size = (5, 5)
        self.kernel = np.ones((3, 3), np.uint8)
        self.distance_scale = 0.35  # For controlling distance transform
        self.bilateral_d = 50
        self.bilateral_ss = 90

        # Segmenter instantiation
        self._segmenter = None
        if isinstance(segmenter, str):
            self.segmenter_class = segmenter_mapper.get(segmenter, None)
            if self.segmenter_class is None:
                raise Exception(
                    f"{segmenter} not mapped to a valid segmenter class.\nUse: {list(segmenter_mapper.keys())}"
                )
        elif isinstance(segmenter, AbstractSegmenter):
            # If this is an instantiated AbstractSegmenter, set it
            self._segmenter = segmenter
        elif issubclass(segmenter, AbstractSegmenter):
            # Assume this is a
            self.segmenter_class = segmenter
        else:
            raise Exception(
                f"{segmenter} is not mappable str, "
                | "an object inheriting AbstractSegmenter, "
                | "or a class that inherits AbstractSegmenter"
            )

        # hidden variables
        self._label_increment = LABEL_INCREMENT
        self._df = None
        self._region_arr = None
        self._region_dict = None

        # Applet variables
        self._region_tracker = None  # Used for keeping tabs on where in region list we are, created in self.df

        # Initialize input path
        self._filename = filename
        self.input_path = self._input_path

    def reset(self):
        """
        Set the class to have no processed images or data
        """
        self._df = None
        self._image_read = None
        self._image_cropped = None
        self._image_working = None
        self._image_labeled = None
        self._thresh = None
        self.segmenter.image = None
        self._region_dict = None
        self._region_arr = None

    @property
    def filename(self):
        if self._filename is None:
            self._filename = "tmp"
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    @property
    def input_path(self):
        return self._input_path

    @input_path.setter
    def input_path(self, value):
        self._input_path = value
        if self._input_path is not None:
            # Redefine internal paths (these may be removed at some point)
            if isinstance(self._input_path, str):
                self.filename = ".".join(self.input_path.split("/")[-1].split(".")[:-1])
            else:
                temp_image = self._input_path
                self._input_path = f"{self.filename}.png"

                cv2.imwrite(
                    self._input_path,
                    temp_image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0],
                )

            # Clear the dataframe and labeled_image, if it exists
            self.reset()

            # Load into the segmenter
            self.segmenter.image = self.image_cropped

            self._csv_file = str(
                Path(self.result_folder_path) / f"values_{ Path(self._filename).stem }_{self.file_str}.csv"
            )

            # self.process_images(edge_modification=self.edge_modification)

    @property
    def segmenter(self):
        if self._segmenter is None:
            self._segmenter = self.segmenter_class(
                self.image_cropped, **self.segmenter_kwargs
            )
        return self._segmenter

    @property
    def image_read(self):
        if self._image_read is None:
            self._image_read = cv2.imread(self.input_path, 0)
        return self._image_read

    @property
    def image_cropped(self):
        if self._image_cropped is None:
            self._image_cropped = self.image_read[
                self.top_boundary : self.bottom_boundary,
                self.left_boundary : self.right_boundary,
            ]
        return self._image_cropped

    @property
    def image_working(self):
        if self._image_working is None:
            self._image_working = cv2.cvtColor(self.image_cropped, cv2.COLOR_GRAY2BGR)
        return self._image_working

    @property
    def image_labeled(self):
        if self._image_labeled is None:
            self.process_images()
        return self._image_labeled

    @property
    def thresh(self):
        return self.segmenter.thresh

    @property
    def markers(self):
        return self.segmenter.markers

    @property
    def markers_filled(self):
        return self.segmenter.markers_filled

    def process_images(self):
        """
        Create each of the internal images of interest.
        Performs segmentation as part of the process
        """
        if self.input_path is None:
            raise Exception("Error: ImageSegmenter has no input_path")

        # Set regions by number, non-inclusive of background and edge border
        self.regions_list = np.unique(self.markers_filled) - self._label_increment
        self.regions_list = [x for x in self.regions_list if x > 0]

        self._image_labeled = color.label2rgb(self.markers_filled, bg_label=0)

    def decorate_regions(self):
        """
        Labels image 4 using information from Scikit to ensure commensurate labeling.
        Generally a useful visualization tool
        NOTE: The big issue is ensuring regions line up
        """

        for i in self.regions_list:
            cx = int(self.outputs[3][i][0])
            cy = int(self.outputs[3][i][1])
            cv2.putText(
                self.image_labeled,
                text=str(i),
                org=(cx, cy),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                self.image_labeled, (cx, cy), radius=3, color=(0, 0, 0), thickness=-1
            )
        return

    @property
    def df(
        self,
    ):
        if self._df is None:
            # Make sure to instantiate the images
            self.image_labeled

            file_present = os.path.isfile(self._csv_file)
            if file_present and not self.override_exists:
                df = pd.read_csv(self._csv_file)
                self.number_labels = len(df["area"])
                self._df = pd.read_csv(self._csv_file)
                self._region_tracker = self._df["Region"].min()
                return self._df

            propList = [
                "area",
                "equivalent_diameter",
                "orientation",
                "major_axis_length",
                "minor_axis_length",
                "perimeter",
                "min_intensity",
                "mean_intensity",
                "max_intensity",
                "solidity",
                "eccentricity",
                "centroid_local",
                "feret_diameter_max",
                "moments",
                "moments_central",
                "moments_hu",
                "label",
            ]
            clusters = measure.regionprops_table(
                self.markers_filled - self._label_increment,
                self.image_cropped,
                properties=propList,
            )

            scaled_features = [
                "equivalent_diameter",
                "major_axis_length",
                "minor_axis_length",
                "perimeter",
                "feret_diameter_max",
                #'solidity'
            ]

            # Apply pixel scaler ot geometric features
            for key, val in clusters.items():
                # print(f'{key}: {len(val)}')
                if key == "area":
                    clusters[key] = clusters[key] * self.pixels_to_um**2
                if key == "orientation":
                    continue  # Line didn't seem used to me previously...?
                if key == "label":
                    continue
                elif key in scaled_features:
                    clusters[key] = clusters[key] * self.pixels_to_um

            # Add in Composite variables
            clusters["major_axis_length/minor_axis_length"] = (
                clusters["major_axis_length"] / clusters["minor_axis_length"]
            )
            clusters["perimeter/major_axis_length"] = (
                clusters["perimeter"] / clusters["major_axis_length"]
            )
            clusters["perimeter/minor_axis_length"] = (
                clusters["perimeter"] / clusters["minor_axis_length"]
            )

            # Add in Label, Filename, Region Columns
            self.number_labels = len(clusters["area"])
            labeling_list = [None] * self.number_labels
            filename_list = [self.input_path] * self.number_labels
            clusters["Labels"] = labeling_list
            clusters["Filename"] = filename_list
            clusters["Region"] = clusters["label"]

            # Create df
            self._df = pd.DataFrame(clusters)
            self._region_tracker = self._df["Region"].min()

            # Add region featurizer info
            if len(self.region_featurizers) > 0:
                region_dict = self.grab_region_dict(
                    self.image_cropped, focused=False, alpha=0
                )

                def row_add_features(row: pd.Series):
                    region_img = region_dict[row.Region]
                    region = feat.Region(
                        region_img, featurizers=self.region_featurizers
                    )
                    return pd.Series(region.featurize())

                df_regions = self._df.apply(row_add_features, axis=1)
                self._df = pd.concat([self._df, df_regions], axis=1)

        return self._df

    def create_csv(self):
        '''
        Simple function for saving csv. Helper function for live labeling to ensure progress is not lost
        '''
        if self.override_exists:
            os.makedirs(self.result_folder_path, exist_ok=True)
            self.df.to_csv(self._csv_file)
        else:
            print("WARNING: Override not in place")

    def to_h5(self, file_name, mode="w"):
        """
        Save all images and regions to an h5 file for easy access
        Since some Regions may be skipped during measurement, need to key on this
        Args:
            file_name (str) : Name of file
            mode (str) : Method by which to access the file
        """

        f = h5py.File(file_name, mode)

        group_name = Path(self._input_path).stem
        group = f.create_group(group_name)

        group.create_dataset("input_path", data=self.input_path)
        group.create_dataset("input_image", data=self.image_read)
        group.create_dataset("image_cropped", data=self.image_cropped)
        group.create_dataset("image_working", data=self.image_working)
        group.create_dataset("image_labeled", data=self.image_labeled)
        group.create_dataset("thresh", data=self.thresh)
        group.create_dataset("markers_filled", data=self.markers_filled)

        # Load in all regions identified
        dset = group.create_dataset(
            "Regions", shape=(self.df.Region.max(), *np.shape(self.markers_filled))
        )
        for ii, (key, region) in enumerate(
            self.grab_region_dict(self.image_cropped, focused=False, alpha=0).items()
        ):
            # Some regions are grabbed in error
            if len(region) == 0:
                continue
            dset[key - 1, :, :] = region  # Subtract by 1 since regions are 1-indexing

        f.close()

    def _grab_region(self, img, region_oi, alpha=0.75, buffer=20):
        label_oi = region_oi + self._label_increment
        mod_image = copy.deepcopy(img)
        label_marker = copy.deepcopy(self.markers_filled)
        label_marker[self.markers_filled != label_oi] = 0

        y1 = grab_bound(label_marker, "top", buffer)
        y2 = grab_bound(label_marker, "bottom", buffer)
        x1 = grab_bound(label_marker, "left", buffer)
        x2 = grab_bound(label_marker, "right", buffer)

        mod_image[label_marker != label_oi] = (
            mod_image[label_marker != label_oi] * alpha
        )
        return mod_image[y1:y2, x1:x2]

    @property
    def region_arr(self):
        """
        img Regions associated with the image segmentation
        NOTE: _Slightly_ different from 'begin_labeling' as we remove non-region ENTIRELY
        """
        if self._region_arr is None:
            self._region_arr = self.grab_region_array(focused=True)
        return self._region_arr

    @property
    def region_dict(self):
        if self._region_dict is None:

            self._region_dict = self.grab_region_dict(focused=True, alpha=0.7)

        return self._region_dict

    def grab_region_array(self, img_oi=None, focused=True, alpha=0, buffer=5):
        """
        Grab an array of images that are bounded (focused) or the same size as image_cropped (not focused)
        Can be useful for quickly making bools of regions of any internal image, so is distinguished from region_arr attribute
        Args:
            img_oi (np.ndarray) : image with same size as working image. Can be markers, image_working, etc.
            focused (bool) : If focused, return regions focused solely on the region area, plus the buffer amount of pixels on each side. 
                            Helpful for visualization
            alpha (float) : Alpha channel for pixels not associatredd with the region. Can highlight difference in region and dnearby image spots
            buffer (int) : Buffer pixels to pad to mask if focusing the image
        """
        if img_oi is None:
            img_oi = self.image_cropped
        self.df  # Make sure this is initiated
        data_arr = []
        ii = 0
        regions_list = list(self.df["Region"])

        while ii < len(regions_list):  # 1-Offset for counting purposes
            region_oi = regions_list[ii]
            if focused:
                data_arr.append(
                    self._grab_region(img_oi, region_oi, alpha=alpha, buffer=buffer)
                )
            if not focused:
                data_arr.append(
                    self._grab_region(img_oi, region_oi, alpha=alpha, buffer=np.inf)
                )
            ii += 1
        return data_arr

    def grab_region_dict(self, img_oi=None, focused=True, alpha=0.7):
        """
        Grab a dict of regions that are bounded (focused) or the same size as image_cropped (not focused)
        Can be useful for quickly making bools of regions of any internal image, so is distinguished from region_arr attribute
        Args:
            img_oi (np.ndarray) : image with same size as working image. Can be markers, image_working, etc.
            focused (bool) : If focused, return regions focused solely on the region area, plus the buffer amount of pixels on each side. 
                            Helpful for visualization
            alpha (float) : Alpha channel for pixels not associatredd with the region. Can highlight difference in region and dnearby image spots
            buffer (int) : Buffer pixels to pad to mask if focusing the image
        """

        if img_oi is None:
            img_oi = self.image_cropped
        self.df  # Make sure this is initiated
        regions_list = list(self.df["Region"])
        data_dict = {}
        for region in regions_list:  # 1-Offset for counting purposes
            region_oi = region
            if focused:
                data_dict[region] = self._grab_region(
                    img_oi, region_oi, alpha=alpha, buffer=15
                )
            if not focused:
                data_dict[region] = self._grab_region(
                    img_oi, region_oi, alpha=alpha, buffer=np.inf
                )
        return data_dict

    def begin_labeling(
        self,
        labeling_dict={
            "C": "Crystal",
            "M": "Multiple Crystal",
            "P": "Poorly Segmented",
            "I": "Incomplete",
        },
    ):
        """
        Major Utility function for labeling of segmented regions in a jupyter notebook.
        Args:
            labeling_dict (dict) : To speed up labeling, assign letters to a full label for easy mapping. Will also catch missed keystrokes
        """
        # Make sure B and D are not overwritten
        if "B" in labeling_dict or "D" in labeling_dict:
            raise Exception("Cannot use 'B' or 'D' in labeling_dict")

        # Develop options
        options_list = labeling_dict.keys()
        options_str = ", ".join(
            [f"{key} = {val}" for key, val in labeling_dict.items()]
        )

        self.df  # To ensure it's been initialized
        ii = 0

        # NOTE: Use this instead of self.region_arr or self.region_dict to avoid overwrite issues
        regions_list = self.df["Region"]
        while ii < len(regions_list):
            clear_output(wait=False)
            region_oi = regions_list[ii]

            testImage = self._grab_region(
                self.image_working,
                region_oi + self._label_increment,
                alpha=0.75,
                buffer=20,
            )
            plt.figure(figsize=(10, 10))
            plt.imshow(testImage)
            plt.show()

            # User Input
            input_list = [*options_list, "B", "D"]

            print(
                f"Region {region_oi} (Max: {max(regions_list)}) \nNOTE: Skipping a region may mean a bad region was encountered\n"
            )
            print(
                "Type an integer to jump to region, or a character below to label image\n",
                options_str,
                "\nB = Back, D = Done",
            )
            user_input = input()
            while (user_input not in input_list) and (not user_input.isnumeric()):
                user_input = input("Invalid Input, Retry: ")
            if user_input == "B":
                ii = ii - 1
                continue
            elif user_input.isnumeric():
                ii = int(user_input) - 1  # Because 1-Offset
                continue
            elif user_input == "D":
                break

            # Clean-up
            translated_input = labeling_dict[user_input]

            # Save for live editing and to not lose information
            self.df.loc[self.df["Region"] == region_oi, "Labels"] = translated_input
            self.df.to_csv(self._csv_file)

            ii = ii + 1

    ## Applet Helper functions below
    def update_df_label_at_region(self, label, region=None):
        '''
        Set label for the dadtaframe row of region
        Args:
            label (str) : Label to store
            region (int) : Region row to target
        '''
        if region is None:
            region = self._region_tracker
        self.df.loc[self.df["Region"] == region, "Labels"] = label

    def labeling_mapping(self):
        """
        Code added 2022.08.12 for debugging and salvaging data. 
        Issue: Row, index, andd posiiton in array were messedd dup by 0 to 1 pixel regions disappearing. This salvaged data
        by recreating the originaal dadta and dthen approrpiately aaccounting for offset.
        Kept for reference 
        """
        self.df  # To ensure it's been initialized
        ii = 0
        regions_list = self.df["Region"]
        mapping_index = []
        mapping_region = []

        while ii < len(regions_list):  # 1-Offset for counting purposes
            clear_output(wait=False)
            region_oi = regions_list[ii]  # +3 gets past Borders and BG labeling

            # self.df[self.df['Region'] == region_oi]['Labels'] = translated_input
            mapping_index.append(ii)
            mapping_region.append(region_oi)
            ii = ii + 1
        return mapping_region, mapping_index



class BatchImageSegmenter:

    def __init__(
        self,
        img_list=None,
        IS_list=None,
        pixels_to_um=9.37,
        top_boundary=0,
        bottom_boundary=860,
        left_boundary=0,
        right_boundary=2560,
        result_folder_path="Results",
        override_exists: bool = True,
        segmenter: str = "algorithmic",
        segmenter_kwargs: dict = {},
        region_featurizers=[
            feat.AverageCurvatureFeaturizer(),
            feat.StdCurvatureFeaturizer(),
            feat.MinCurvatureFeaturizer(),
            feat.MaxCurvatureFeaturizer(),
            feat.PercentConvexityCurvatureFeaturizer(),
            feat.LongestContiguousConcavityCurvatureFeaturizer(),
            feat.LongestContiguousConvexityCurvatureFeaturizer(),
            feat.DistinctPathsCurvatureFeaturizer(),
        ],
        file_str=None,
        filename_list=None,
    ):
        """
        Class for doing batch processing of an image segmenter.
        Compared to a regular ImageSegmenter, all functional and proeprty calls here simply grab
         and concatenate the individual ImageSegmenters together.

         Use this in cases where holding all images simultaneously is desirable,
         but be warned it can take in a large amount of memory!

         This is used internally for the applet
        """
        self.pixels_to_um = pixels_to_um
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.result_folder_path = result_folder_path
        self.override_exists = override_exists
        self.segmenter = segmenter
        self.segmenter_kwargs = segmenter_kwargs
        self.file_str = file_str
        self.filename_list = filename_list

        # Template
        self._template_IS = ImageSegmenter(
            pixels_to_um=self.pixels_to_um,
            top_boundary=self.top_boundary,
            bottom_boundary=self.bottom_boundary,
            left_boundary=self.left_boundary,
            right_boundary=self.right_boundary,
            result_folder_path=self.result_folder_path,
            override_exists=self.override_exists,
            segmenter=self.segmenter,
            segmenter_kwargs=self.segmenter_kwargs,
            region_featurizers=region_featurizers,
            file_str=self.file_str,
        )

        # Input values
        self._img_list = None
        self._IS_list = None  # Try to keep these updated in parallel w/ eachother

        if img_list:
            self.img_list = img_list
        if IS_list:
            self.IS_list = IS_list

        # pyqt helpers, will access the ImageSegmenters together
        self._df = None
        self._region_arr = None
        self._region_dict = None
        self._region_tracker = None

        self._batch_region_dict = None
        self._IS_index = None  # Easiest way to check which ImageSegmenter we're in

        # Template ImageSegmenter (for if images are appended AFTER)

    @property
    def img_list(self):
        return self._img_list

    @img_list.setter
    def img_list(self, val):
        self._img_list = val

        self._IS_list = []
        for ii, img_oi in enumerate(self._img_list):
            ready_IS = copy.deepcopy(self._template_IS)

            # Create an IS
            if self.filename_list:
                assert len(self.filename_list) == len(self._img_list)
                ready_IS.filename = self.filename_list[ii].split(".")[0]
            ready_IS.input_path = img_oi
            self._IS_list.append(ready_IS)

    @property
    def IS_list(self):
        return self._IS_list

    @IS_list.setter
    def IS_list(self, val):
        self._IS_list = val

        self._img_list = []
        for IS in val:
            self._img_list.append(IS._input_path)

    def __getitem__(self, index):
        return self.IS_list[index]

    def __setitem__(self, index, newValue):
        if isinstance(newValue, ImageSegmenter):
            self._IS_list[index] = newValue
            self._img_list[index] = newValue.input_path
        else:
            self._img_list[index] = newValue
            ready_IS = copy.deepcopy(self._template_IS)
            ready_IS.input_path = newValue
            self._IS_list[index] = ready_IS

    def append(self, val):
        if isinstance(val, ImageSegmenter):
            self._IS_list.append(val)
            self._img_list.append(val.input_path)
        else:
            self._img_list.append(val)
            ready_IS = copy.deepcopy(self._template_IS)
            ready_IS.input_path = val
            self._IS_list.append(ready_IS)

    @property
    def df(self):
        """
        Access the dataframe of EVERY ImageSegmenter here by concatting them
        """
        self._df = pd.concat([IS.df for IS in self._IS_list])
        # if self._df is None:
        #    self._df = pd.concat([IS.df for IS in self._IS_list])
        return self._df

    @property
    def region_arr(self):
        if self._region_arr is None:
            self._region_arr = []
            for IS in self._IS_list:
                self._region_arr.extend(IS.region_arr)

        return self._region_arr

    @property
    def region_dict(self):
        if self._region_dict is None:
            self._batch_region_dict = BatchedRegionDict(
                [IS.region_dict for IS in self._IS_list]
            )
            self._region_dict = self._batch_region_dict

        return self._region_dict


# Need to define a batched region class to work
class BatchedRegionDict:
    def __init__(self, list_of_dicts):
        """
        Dict-like class that only supports getting items
        Should make it easier to grab regions
        """
        self.grouped_dict = {ii: dict_oi for ii, dict_oi in enumerate(list_of_dicts)}

    def __getitem__(self, val):
        val_tracker = val
        for key, item in self.grouped_dict.items():
            check_inside = val_tracker - len(item)

            if check_inside < 0:
                # Must be inside this current item
                return item[val_tracker]
            else:
                val_tracker = check_inside

        raise Exception("BatcheddRegionDict Exception: Out of range")

    def __setitem__(self, val):
        raise Exception("BatchedRegionDict Error: Setting values not supported")

    def __len__(self):
        return np.sum([len(ii) for _, ii in self.grouped_dict.items()])


def grab_bound(img, mode="top", buffer=0):
    """
    For an intensity img with region of interest and all others blacked out, get a bound defined by mode

    Returns x- or y-coordinate for the dedisgnated 'mode'
    Args:
        moded (str) : 'top', 'bottom', 'left', or 'right'
    """

    def bounded_expansion(coord, img, axis):
        if coord < 0:
            return 0
        elif coord > np.shape(img)[axis]:
            return np.shape(img)[axis]
        else:
            return coord

    if mode == "top":
        for yy in np.arange(0, np.shape(img)[0]):
            num_list = np.unique(img[yy, :])
            if len(num_list) > 1:
                return bounded_expansion(yy - buffer, img, 0)

    elif mode == "bottom":
        for yy in np.arange(0, np.shape(img)[0])[::-1]:
            num_list = np.unique(img[yy, :])
            if len(num_list) > 1:
                return bounded_expansion(yy + buffer, img, 0)

    elif mode == "left":
        for xx in np.arange(0, np.shape(img)[1]):
            num_list = np.unique(img[:, xx])
            if len(num_list) > 1:
                return bounded_expansion(xx - buffer, img, 1)

    elif mode == "right":
        for xx in np.arange(0, np.shape(img)[1])[::-1]:
            num_list = np.unique(img[:, xx])
            if len(num_list) > 1:
                return bounded_expansion(xx + buffer, img, 1)
    return -1
