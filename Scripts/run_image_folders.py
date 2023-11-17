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
threshold_mode = "ensemble"
IS = ImageSegmenter(input_path=None,
               threshold_mode=threshold_mode,
               edge_modification="localthresh",)

## Define Images ##

parent_image_dir = os.path.join(Path(__file__).parent.parent,"Images","2023_11_Organized_Images")

parent_image_dir = os.path.join(Path(__file__).parent.parent,"Images","Fig_3_PAE_diffusion")
image_folders = [path for path in glob.glob(os.path.join(parent_image_dir,"*")) if os.path.isdir(path)]

## Define Save Path
results_path = f"../Results/2023_11_Organized_Images_{threshold_mode}"

## Load Features
features = load_feature_config("default_features-agnostic")

## Load Models
model_set = "2023_original_default_features-agnostic"
model_folder = os.path.join(Path(__file__).parent.parent,"facet_ml","static","Models",model_set)
model_CvMC_path = os.path.join(model_folder,"RF_C_MC.sav")
model_CvI_path = os.path.join(model_folder,"RF_C-MC_I.sav")
with open(model_CvMC_path,"rb") as f:
    model_CvMC = pickle.load(f)
with open(model_CvI_path,"rb") as f:
    model_CvI = pickle.load(f)



if __name__ == "__main__":
    for image_folder in image_folders:
        image_list = glob.glob(os.path.join(image_folder,"*"))
        _,image_folder_name = os.path.split(image_folder)
        pbar = tqdm.tqdm(image_list)
        for image in pbar:
            image_name = Path(image).stem
            pbar.set_description(f"{image_folder_name,image_name}")
            IS.input_path = image
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
            csv_path = os.path.join(save_folder,f"{image_name}+_RESULTS.csv")
            img_path = os.path.join(save_folder,f"{image_name}+_VISUALZIATION.png")

            # Save
            cv2.imwrite(img_path,cv2.cvtColor(color_img,cv2.COLOR_BGR2RGB))
            IS.df.to_csv(csv_path)