# Script will accept a an image folder that contains
# sub-image folders and run through them. Data will be collected

import sys

sys.path.append("..")

from facet_ml.classification.model_using import *
from facet_ml.segmentation.segmenter import ImageSegmenter
from facet_ml.classification.model_training import load_feature_config
import os
import pickle
import numpy as np
import cv2
import copy
import glob
import tqdm
from pathlib import Path

## Define Image Segmenter run conditions ##
# threshold_mode = "segment_anything"
# IS = ImageSegmenter(input_path=None,
#                threshold_mode=threshold_mode,
#                edge_modification="localthresh",)

threshold_mode = "segment_anything"
segmenter_kwargs = {
    "threshold_mode": "segment_anything",
    "edge_modification": "localthresh",
}
IS = ImageSegmenter(
    input_path=None,
    segmenter=threshold_mode,
    # segmenter_kwargs=segmenter_kwargs,
)

## Target Image
image_path = r"C:\Users\Jacob\Desktop\Academics\Mirkin\colloidal_crystal_ML\Images\20240217_Images_by_experiment\Fig3_Fig4_oven_pcr_linkerstrength\L-2_nM-2.5_mixing-F_oven-F_embed-Ag\L2_2.5nM_03.bmp"

## Load Features
# features = load_feature_config("default_features-agnostic")
# features = load_feature_config("default_features")
features = load_feature_config("2024_features-agnostic")

## Load Models
model_set = "2024_02__original_default_features-agnostic"
# model_set = "2024_02__original_default_features"
# model_set = "2023_original_default_features-agnostic"  # Best sent RC's way
model_set = "2024_02__original_2024_features-agnostic"
model_folder = os.path.join(
    Path(__file__).parent.parent, "facet_ml", "static", "Models", model_set
)
model_CvMC_path = os.path.join(model_folder, "RF_C_MC.sav")
model_CvI_path = os.path.join(model_folder, "RF_C-MC_I.sav")
with open(model_CvMC_path, "rb") as f:
    model_CvMC = pickle.load(f)
    if isinstance(model_CvMC, dict):
        model_CvMC = model_CvMC["model"]
with open(model_CvI_path, "rb") as f:
    model_CvI = pickle.load(f)
    if isinstance(model_CvI, dict):
        model_CvI = model_CvI["model"]

### MAIN BODY OF CODE ###

# if __name__ == "__main__":

## Define Save Path
image_name = Path(image_path).stem
IS.input_path = image_path
# IS.process_images()
MA_CvMC = ModelApplication(model_CvMC, IS, features=features)
MA_CvI = ModelApplication(
    model_CvI,
    IS,
    features=features,
    replacement_dict={
        "Crystalline": MA_CvMC,
        "Not Crystalline": "Incomplete",
    },
)

IS.df["Labels"] = MA_CvI.run()
color_img = visualize_labels(IS, IS.df)

# Prepare to save
# save_folder = os.path.join(results_path, image_folder_name)
# os.makedirs(save_folder, exist_ok=True)
csv_path = os.path.join(f"{image_name}_results.csv")
img_path = os.path.join(f"{image_name}_visualiation.png")

# Save
cv2.imwrite(img_path, cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
IS.df.to_csv(csv_path)
