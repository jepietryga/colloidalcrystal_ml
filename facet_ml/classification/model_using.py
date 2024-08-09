from typing import Union
from facet_ml.segmentation.segmenter import ImageSegmenter
from facet_ml.classification.model_training import load_feature_config
import numpy as np
from typing import Union, Dict
import pandas as pd
import cv2
import copy


class ModelApplication:

    def __init__(
        self,
        model: Union[callable, str],
        image_segmenter: ImageSegmenter,
        features: Union[str, list],
        targets: list[str] = "Labels",
        replacement_dict: Dict[str, object] = {},
        featurizers: list = [],
        image: Union[np.ndarray, str] = None,
    ):
        """
        Class for wrapping the application of a model to an ImageSegmenter

        replacement dict will hold
        {"<LABEL>":Model2 | <NEW_LABEL>}
        self.image_segmenter.df == "<LABEL>" is replaced by results from Model2 OR <NEW_LABEL>
        This allows creation of model trees
        """
        self.model = model
        self.image_segmenter = image_segmenter
        self._image = image
        self.features = self.features = (
            features if isinstance(features, list) else load_feature_config(features)
        )
        self.targets = targets
        self.replacement_dict = replacement_dict
        self.featurizers = featurizers

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value
        self.image_segmenter.input_path = self._image

    def recursive_replacement(self, df):
        """
        Use the replacement_dict models to replace results in the given dataframe
        NOTE:
        """

        for label, replacer in self.replacement_dict.items():
            index = df[df[self.targets] == label].index
            if isinstance(replacer, str):
                df.loc[index, self.targets] = replacer
            else:
                recursive_result = replacer.run()
                df.loc[index, self.targets] = recursive_result[index]

    def run(self) -> pd.Series:
        if isinstance(self.image_segmenter.image_working, type(None)):
            raise Exception("ImageSegmenter does not have a valid image")

        for featurizer in self.featurizers:
            featurizer(self.image_segmenter)
        df_working = self.image_segmenter.df.copy()

        df_working = df_working[self.features]

        # Replace nans
        df_working.replace([np.inf, -np.inf], 0, inplace=True)
        df_working.fillna(0, inplace=True)

        df_working[self.targets] = self.model.predict(df_working)
        self.recursive_replacement(df_working)
        return df_working[self.targets]


def visualize_labels(
    IS: ImageSegmenter,
    df: pd.DataFrame,
    color_dict: dict = {
        "Crystal": np.array([0, 0, 255]),
        "Multiple Crystal": np.array([0, 100, 0]),
        "Incomplete": np.array([255, 0, 0]),
        "Poorly Segmented": np.array([0, 255, 255]),
    },
    default_color: np.ndarray = np.array([255, 0, 255]),
):
    """
    Given an ImageSegmenter and a dataframe post-processing, apply colors
    based on labels (color_dict)to the image.
    """
    region_arr = IS.grab_region_array(focused=False)
    mod_image = cv2.cvtColor(IS.image_cropped, cv2.COLOR_BGR2RGB)
    mask_image = copy.deepcopy(mod_image) * 0
    ii = 0
    for index, row in df.iterrows():
        id_label = row["Labels"]
        color = color_dict.get(id_label, np.array([255, 0, 255]))
        mask_logical = region_arr[ii] > 0
        edge_logical = dilate_logical(mask_logical, 2)
        mask_image[edge_logical] = np.array([255, 255, 255])
        mask_image[mask_logical] = color
        ii += 1

    final_image = cv2.addWeighted(mod_image, 1, mask_image, 0.5, 0)
    return final_image


def dilate_logical(array: np.ndarray, dilate_size: int = 1) -> np.ndarray:
    """
    Given Boolean logical array, dilate it by vectorizing rolling a stack of images
    """

    return_array = array.copy()

    # Rolling
    for ii in range(dilate_size):
        roll_ref = return_array.copy()
        roll_size = ii + 1
        return_array = (
            np.roll(roll_ref, (0, roll_size))
            | np.roll(roll_ref, (0, -roll_size))
            | np.roll(roll_ref, (roll_size, 0))
            | np.roll(roll_ref, (-roll_size, 0))
            | roll_ref
        )

    return return_array
