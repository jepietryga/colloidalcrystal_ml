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

threshold_mode = "algorithmic"
segmenter_kwargs = {
    "edge_modification":"localthresh"
}
IS = ImageSegmenter(
    input_path=None,
    segmenter=threshold_mode,
    segmenter_kwargs=segmenter_kwargs,
)
## Define Images ##
'''
parent_image_dir = os.path.join(Path(__file__).parent.parent,"Images","2023_11_Organized_Images")

parent_image_dir = os.path.join(Path(__file__).parent.parent,"Images","Fig_3_PAE_diffusion")
parent_image_dir = os.path.join(Path(__file__).parent.parent,"Images","2023_12_02_Figure2_Images")
parent_image_dir = os.path.join(Path(__file__).parent.parent,"Images","2024_01_03_Request")
parent_image_dir = os.path.join(Path(__file__).parent.parent,"Images","Fig_3_PAE_diffusion")
parent_image_dir = Path(__file__).parent.parent / "Images" / "2024_02_Fig_4_rate_images"

parent_image_dir_list = []
parent_image_dir_list.append(Path(__file__).parent.parent / "Images" / "20240217_Images_by_experiment")

#image_folders = [path for path in glob.glob(os.path.join(parent_image_dir,"*")) if os.path.isdir(path)]

## Define Save Path
results_path = f"../Results/2024_02_Figure4_rate_images_{threshold_mode}"
'''

## Target folders
parent_image_dir_list = []
# parent_image_dir_list.append(Path(__file__).parent.parent / "Images" / "20240217_Images_by_experiment" / "Fig3_Fig4_oven_pcr_linkerstrength")
# parent_image_dir_list.append(Path(__file__).parent.parent / "Images" / "20240217_Images_by_experiment" / "Fig4_slow_cooling_rates")
parent_image_dir_list.append(Path(__file__).parent.parent / "Images" / "Fig_3_PAE_diffusion")

## Load Features
features = load_feature_config("default_features-agnostic")
features = load_feature_config("2024_features-agnostic")

## Load Models
model_set = "2024_02__original_default_features-agnostic"
model_set = "2024_02__original_2024_features-agnostic"
model_folder = os.path.join(Path(__file__).parent.parent,"facet_ml","static","Models",model_set)
model_CvMC_path = os.path.join(model_folder,"RF_C_MC.sav")
model_CvI_path = os.path.join(model_folder,"RF_C-MC_I.sav")
with open(model_CvMC_path,"rb") as f:
    model_CvMC = pickle.load(f)
    if isinstance(model_CvMC,dict):
        model_CvMC = model_CvMC["model"]
with open(model_CvI_path,"rb") as f:
    model_CvI = pickle.load(f)
    if isinstance(model_CvI,dict):
        model_CvI = model_CvI["model"]

### MAIN BODY OF CODE ###

#if __name__ == "__main__":
for parent_image_dir in parent_image_dir_list:
    image_folders = [path for path in glob.glob(os.path.join(parent_image_dir,"*")) if os.path.isdir(path)]

    ## Define Save Path
    save_id = Path(parent_image_dir).stem
    results_path = f"../Results/{save_id}_{threshold_mode}"

    for image_folder in image_folders:
        image_list = glob.glob(os.path.join(image_folder,"*"))
        _,image_folder_name = os.path.split(image_folder)
        pbar = tqdm.tqdm(image_list)
        for image in pbar:
            image_name = Path(image).stem
            pbar.set_description(f"{image_folder_name,image_name}")
            IS.input_path = image
            IS.process_images()
            MA_CvMC = ModelApplication(model_CvMC,IS,features=features)
            MA_CvI = ModelApplication(model_CvI,IS,features=features,
                                      replacement_dict={
                                          "Crystalline":MA_CvMC,
                                          "Not Crystalline":"Incomplete"
                                      })
            
            IS.df["Labels"] = MA_CvI.run()
            color_img = visualize_labels(IS,IS.df)

            # Prepare to save
            save_folder = os.path.join(results_path,image_folder_name)
            os.makedirs(save_folder,exist_ok=True)
            csv_path = os.path.join(save_folder,f"{image_name}_results.csv")
            img_path = os.path.join(save_folder,f"{image_name}_visualiation.png")

            # Save
            cv2.imwrite(img_path,cv2.cvtColor(color_img,cv2.COLOR_BGR2RGB))
            IS.df.to_csv(csv_path)
        
        # combine all csvs
        csv_paths = glob.glob(os.path.join(save_folder,"*.csv"))
        df_list = list(map(pd.read_csv,csv_paths))
        df_total = pd.concat(df_list)
        df_total.to_csv( os.path.join(save_folder,f"{image_folder_name}_total_results.csv"))