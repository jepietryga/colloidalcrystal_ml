# Goal of this script is to make several variations of 
# models for easier cross checking of performance
import sys
sys.path.append("..")
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier

from facet_ml.segmentation.segmenter import ImageSegmenter
from facet_ml.classification.model_training import *
from facet_ml.classification.model_using import *
from facet_ml.segmentation.features import facet_score
import model_utils
import cv2

## Define path to folder of csv files ##
## Model grid 
#           |OLD MODEL | New Model (default) | (default-agnostic) | (default-facet_score) | (default-facet_score-agnostic) 
# All-in-One|
# Multi-step|

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
        color = color_arr[id_label]
        mask_image[region_arr[ii] > 0] = color
        ii +=1
    
    final_image = cv2.addWeighted(mod_image,1,mask_image,.5,0)
    return final_image  

def apply_all_in_one(model_folder:str,
                     IS:ImageSegmenter,
                     features):
    model_path = os.path.join(model_folder,"RF_C_MC_I.sav")
    with open(model_path,"rb") as f:
        model = pickle.load(f)
        if isinstance(model,dict):
            model = model["model"]
    MA = ModelApplication(model,
                     image_segmenter=IS,
                     features=features,
                     featurizers=[facet_score])
    
    return MA.run()

def apply_multilevel(model_folder:str,
                     IS:ImageSegmenter,
                     features):
    # Load first model
    model_path_CvNC = os.path.join(model_folder,"RF_C-MC_I.sav")
    with open(model_path_CvNC,"rb") as f:
        model_CvNC = pickle.load(f)
        if isinstance(model_CvNC,dict):
            model_CvNC = model_CvNC["model"]
    
    # Load second model
    model_path_CvMC = os.path.join(model_folder,"RF_C_MC.sav")
    with open(model_path_CvMC,"rb") as f:
        model_CvMC = pickle.load(f)
        if isinstance(model_CvMC,dict):
            model_CvMC = model_CvMC["model"]
    
    # Apply first model
    # REALIZATION: Don't need labels, just drop what ISN'T expected!
    MA_CvNC = ModelApplication(model_CvNC,
                     image_segmenter=IS,
                     features=features,
                     featurizers=[facet_score])
    MA_CvMC = ModelApplication(model_CvMC,
                     image_segmenter=IS,
                     features=features,
                     featurizers=[facet_score])

    df_first = IS.df.copy()
    df_second = IS.df.copy()
    
    df_first["Labels"] = MA_CvNC.run()
    df_second["Labels"] = MA_CvMC.run()
    
    # Merge on first level "Crystalline" label
    crystalline_logical = df_first["Labels"] == "Crystalline" 
    df_first[crystalline_logical] = df_second[crystalline_logical]

    df_first[~crystalline_logical] = "Incomplete"

    return df_first["Labels"]


image_list = ["/Users/jacobpietryga/Desktop/Academics/colloidal_crystal_ML/Images/Training/4 nM 1.bmp"]
# Name:(folder, features in config)
columns_dict = {
    "OLD MODELS":("model_set__original_2023_classifier","default_features"),
    "default_features":("2023_original_default_features","default_features"),
    "default_features-agnostic":("2023_original_default_features-agnostic","default_features-agnostic"),
    "default_features-facet_score":("2023_original_default_features-facet_score","default_features-facet_score"),
    "default_features-facet_score-agnostic":("2023_original_default_features-facet_score-agnostic","default_features-facet_score-agnostic")
}

def make_image_comparison_plot(image,
                               IS_kwargs,
    ):
    fig,ax = plt.subplots(
                        nrows=2,
                        ncols=len(columns_dict),
                        #dpi=300,
                        figsize=(8*len(columns_dict),8),
                        )
    for jj,(model_run_function,run_name) in enumerate([(apply_all_in_one,"All in One"),
                                            (apply_multilevel,"Multilevel")]):
        ax_row = ax[jj,:]
        ax_row[0].set_ylabel(run_name)
        for ii,(name, (folder_oi,features)) in enumerate(columns_dict.items()):
            ax_oi = ax_row[ii]
            IS = ImageSegmenter(image,**IS_kwargs)
            full_folder_path = os.path.join("../facet_ml/static/Models",folder_oi)
            IS.df["Labels"] = results = model_run_function(full_folder_path,IS,features)

            im = apply_coloring(IS,IS.df)
            ax_oi.imshow(im)
            ax_oi.set_title(name)
    return fig


if __name__ == "__main__":
    image_folder = "/Users/jacobpietryga/Desktop/Academics/colloidal_crystal_ML/Images/Training"
    image_paths = glob.glob(os.path.join(image_folder,"*"))
    for image in image_paths:
        folder_id = image_folder.split("/")[-1]
        image_id = ".".join(
                    image.split("/")[-1].split(".")[:-1]
                )
        IS_kwargs = {

            "threshold_mode":"segment_anything",
            "edge_modification":"localthresh"
        }
        figure = make_image_comparison_plot(image,
                                            IS_kwargs)

        os.makedirs(folder_id,exist_ok=True)
        save_name = os.path.join(folder_id,image_id+".png")
        figure.savefig(save_name)
