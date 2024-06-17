## (2024.06.05)
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
import json

## Define Image Segmenter run conditions ##
threshold_mode = "segment_anything"
segmenter_kwargs = {
    # "edge_modification":"localthresh"
}
IS = ImageSegmenter(
    input_path=None,
    segmenter=threshold_mode,
    segmenter_kwargs=segmenter_kwargs,
)

## Define Images ##
IMAGE_FOLDER = Path(__file__).parent.parent / "Image_Data_Repository"

## Folders to Run ##
# Grab based on keyword check
keyword_oi = "sintered"
with open(IMAGE_FOLDER / "folder_information.json", "r") as f:
    folder_info = json.load(f)

folders_to_run = [key for key,items in folder_info.items() 
                    if keyword_oi in items["additional_labels"]
                ]
folder_paths = [str(IMAGE_FOLDER / f) for f in folders_to_run]

## Load Features
features = load_feature_config(
    "2024_features-agnostic"
)  # load_feature_config("default_features-agnostic")


## Load Models
def get_model_from_path(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
        if isinstance(model, dict):
            model = model["model"]
    return model

model_set = "2024_02__original_2024_features-agnostic"  # "2024_02__original_default_features-agnostic"
model_folder = os.path.join(
    Path(__file__).parent.parent, "facet_ml", "static", "Models", model_set
)
model_CvMC_path = os.path.join(model_folder, "RF_C_MC.sav")
model_CvI_path = os.path.join(model_folder, "RF_C-MC_I.sav")
model_CvMC = get_model_from_path(model_CvMC_path)
model_CvI = get_model_from_path(model_CvI_path)

###### MAIN BODY OF CODE #######
    
for image_folder in folder_paths:
    image_path = Path(image_folder)
    results_path = f"../Results/{image_path.stem}_{threshold_mode}"
    image_list = image_path.glob("*")
    image_list = [i for i in image_list 
                if any([v in str(i) for v in ["png","tif","bmp"]])
                ]
    _, image_folder_name = os.path.split(image_folder)
    pbar = tqdm.tqdm(image_list)
    for image in pbar:
        image_name = Path(image).stem
        pbar.set_description(f"{image_folder_name,image_name}")
        IS.input_path = str(image)
        IS.process_images()
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
        save_folder = os.path.join(results_path, image_folder_name)
        os.makedirs(save_folder, exist_ok=True)
        csv_path = os.path.join(save_folder, f"{image_name}_results.csv")
        img_path = os.path.join(save_folder, f"{image_name}_visualization.png")

        # Save
        cv2.imwrite(img_path, cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        IS.df.to_csv(csv_path)

    # combine all csvs
    csv_paths = glob.glob(os.path.join(save_folder, "*.csv"))
    df_list = list(map(pd.read_csv, csv_paths))
    df_total = pd.concat(df_list)
    df_total.to_csv(
        os.path.join(save_folder, f"{image_folder_name}_total_results.csv")
    )
