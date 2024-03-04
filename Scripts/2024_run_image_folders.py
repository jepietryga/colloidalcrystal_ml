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
threshold_mode = "segment_anything"
IS = ImageSegmenter(input_path=None,
               threshold_mode=threshold_mode,
               edge_modification="localthresh",)

## Define Images ##
IMAGE_FOLDER = Path(__file__).parent.parent / "Images"

## Target folders
parent_image_dir_list = ["/Users/jacobpietryga/Desktop/Academics/colloidal_crystal_ML/Images/Diagnostic_Images"]
#parent_image_dir_list.append(IMAGE_FOLDER / 'Diagnostic_Images' / 'Model_check-diagnostic_images_best')
#parent_image_dir_list.append(Path(__file__).parent.parent / "Images" / "20240217_Images_by_experiment" / "Fig3_Fig4_oven_pcr_linkerstrength")
#parent_image_dir_list.append(Path(__file__).parent.parent / "Images" / "20240217_Images_by_experiment" / "Fig4_slow_cooling_rates")

## Load Features
features = load_feature_config("2024_features-agnostic")#load_feature_config("default_features-agnostic")

## Load Models
def get_model_from_path(path:str):
    with open(path,"rb") as f:
        model = pickle.load(f)
        if isinstance(model,dict):
            model = model["model"]
    return model

model_set = "2024_02__original_2024_features-agnostic" #"2024_02__original_default_features-agnostic"
model_folder = os.path.join(Path(__file__).parent.parent,"facet_ml","static","Models",model_set)
model_CvMC_path = os.path.join(model_folder,"RF_C_MC.sav")
model_CvI_path = os.path.join(model_folder,"RF_C-MC_I.sav")
model_CvMC = get_model_from_path(model_CvMC_path)
model_CvI = get_model_from_path(model_CvI_path)

### MAIN BODY OF CODE ###


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