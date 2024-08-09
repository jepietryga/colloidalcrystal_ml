# Script will accept a an image folder that contains 
# sub-image folders and run through them. Data will be collected

import sys
sys.path.append("..")

from facet_ml.classification.model_using import ModelApplication
from facet_ml.segmentation.segmenter import ImageSegmenter
from facet_ml.classification.model_training import load_feature_config
import os
import pickle
import numpy as np
import cv2
import copy

def apply_coloring(IS,df_labeled):
    '''
    To aid study of an image, apply a colored filter over image such that we can see which regions
    are classified as which
    '''
    C_color = np.array([0,0,255])
    MC_color = np.array([255,255,0])
    I_color = np.array([255,0,0])
    P_color = np.array([0,255,0])

    color_arr = {
        "Crystal":C_color,
        "Multiple Crystal":MC_color,
        "Incomplete":I_color,
        "Poorly Segmented":P_color
    }
    match_arr = ["Crystal","Multiple Crystal", "Incomplete", "Poorly Segmented"]
    
    region_arr = IS.grab_region_array(focused=False)
    mod_image = cv2.cvtColor(IS.image_cropped,cv2.COLOR_BGR2RGB)
    mask_image = copy.deepcopy(mod_image)*0
    ii = 0
    for index,row in df_labeled.iterrows():
        id_label = row["Labels"]
        print(row["Labels"])
        color = color_arr.get(id_label,np.array([255,0,255]))
        mask_image[region_arr[ii] > 0] = color
        ii +=1
    
    final_image = cv2.addWeighted(mod_image,1,mask_image,.5,0)
    return final_image 

## Quick sanity checks
model_folder = "../facet_ml/static/Models/2023_original_default_features-agnostic"
model_1_path = os.path.join(model_folder,"RF_C-MC_I.sav")
model_2_path = os.path.join(model_folder,"RF_C_MC.sav")

with open(model_1_path,"rb") as f:
    model_1 = pickle.load(f)

with open(model_2_path,"rb") as f:
    model_2 = pickle.load(f)

image_path = "../Images/Training/4 nM 1.bmp"

IS = ImageSegmenter(input_path=image_path)
features = load_feature_config("default_features-agnostic")
MA_2 = ModelApplication(model_2,image_segmenter=IS,
                        features=features,
                        )

MA_1 = ModelApplication(model_1,IS,features=features,
                        replacement_dict={"Crystalline":MA_2,
                                          "Not Crystalline":"Incomplete"})

IS.df["Labels"] = MA_1.run()

im = apply_coloring(IS,IS.df)
import matplotlib.pyplot as plt
plt.imshow(im)
plt.savefig("plt_test.png")
cv2.imwrite("test.png",im)

