# Purpose of this code is to take in new data and process it

import sys
sys.path.append("..")
import pickle
import os
from sklearn.ensemble import RandomForestRegressor 
import pandas as pd
#from Utility.segmentation_utils import *
from facet_ml.segmentation.segmenter import *
from facet_ml.segmentation import thresholding as th
from facet_ml.segmentation import features as sf
from facet_ml.classification.model_training import *
import tqdm
import matplotlib.pyplot as plt

# SUPPRESS WARNINGS
import warnings
warnings.filterwarnings("ignore")
# Define key helper functions
def assign_label(predicted_data,mode="C-MC_I-P"):
    '''
    Given an array of arrays, get the max column, associate that with a name, and return the fully labeled list
    Should work with 3 given modes
    '''
    valid_modes = ["C-MC_I-P","C_MC","I_P"]
    if mode not in valid_modes:
        print(f'Error: {mode} not supported')
        return -1
    label_arr = []
    for data in predicted_data:
        index = np.argmax(data)
        if mode == valid_modes[0]:
            if index == 0:
                label_arr.append("Crystal")
            if index == 1:
                label_arr.append("Incomplete")
                
        elif mode == valid_modes[1]:
            if index == 0:
                label_arr.append("Crystal")
            if index == 1:
                label_arr.append("Multiple Crystal")
                
        elif mode == valid_modes[2]:
            if index == 0:
                label_arr.append("Incomplete")
            if index == 1:
                label_arr.append("Poorly Segmented")
    return label_arr

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

# Define folders and features in models
model_folder = "../facet_ml/static/Models"+"/2023_11_models"
image_folder = "../Images"
result_folder = "../Results"

features = load_feature_config("default_features-facet_score")


# Load models
model_names = ["RF_C-MC_I-P.sav",
               "RF_C_MC.sav",
]
rf_CMC_IP, rf_C_MC = [pickle.load(open(os.path.join(model_folder,model), 'rb'))\
                              for model in model_names]

# Grab image folder
super_folder = glob.glob("../Images/Figure2_Images_Plus/*")
super_folder = glob.glob("../Images/slow_cooling_rates/*")
super_folder = glob.glob("/Users/jacobpietryga/Desktop/Academics/colloidal_crystal_ML/Images/20230727_RC_Images/*")
#super_folder = glob.glob("../Images/Training/")
image_dir = super_folder[0]

# Get all images
ii = 0
naming_model = "ensemble"
for image_dir in tqdm.tqdm(super_folder):
    print(f"Looking in {image_dir}")
    df_experiment = pd.DataFrame()
    # define subdir
    subdir = os.path.join(result_folder,image_dir.split("/")[-1]+"_ensemble")
    os.makedirs(subdir,exist_ok=True)
    
    ## Grab image
    image_paths = glob.glob(os.path.join(image_dir,"*"))
    for image_path in tqdm.tqdm(image_paths):
        # define subdir
        threshold_mode =[th.otsu_threshold,
                        th.local_threshold,
                        th.pixel_threshold]
                            
        IS = ImageSegmenter(image_path,
                            #threshold_mode="segment_anything",
                            threshold_mode=threshold_mode,
                            edge_modification="localthresh")
        sf.facet_score(IS)
        df_image = IS.df
        

        # Numerical errors (divide by 0)
        df_image.replace([np.inf, -np.inf], np.nan, inplace=True)
        for feature in features:
            df_image.dropna(subset=[feature],inplace=True)
            
        ### Split Crystal & Multicrystal from Incomplete & Poorly Segmented###
        # Split Data
        X=df_image[features]

        predicted_data = rf_CMC_IP.predict(X)
        labeled_arr = assign_label(predicted_data)
        df_image['Labels'] = labeled_arr
        #print("BIG SEPARATOR")
        #print(np.unique(labeled_arr))
        
        #### Subdividing further ####
        df_image_sublist = []
        for label in ['Crystal']:
            df_image_temp = df_image[df_image['Labels'] == label]
            rf_temp = rf_C_MC if label == 'Crystal' else None
            label_mode = "C_MC" if label == 'Crystal' else None
            X_temp = df_image_temp[features]

            try:
                predicted_data_temp = rf_temp.predict(X_temp)
            except:
                print(f'Prediction failed on {label}, maybe 0 elements?')
                predicted_data_temp = []
            labeled_arr = assign_label(predicted_data_temp,mode=label_mode)
            df_image_temp['Labels'] = labeled_arr
            #print(np.unique(df_image_temp["Labels"]))
            df_image_sublist.append(df_image_temp)

        df_image = pd.concat(df_image_sublist)
        df_image.sort_values(by="Region",inplace=True)

        im = apply_coloring(IS,df_image)
        fig, ax = plt.subplots(3,2,dpi=300)
        ax[0,0].imshow(IS.thresh)
        if IS._edge_highlight is not None:
            ax[0,1].imshow(IS._edge_highlight)
        
        ax[1,0].imshow(IS.markers)
        ax[1,1].imshow(IS.markers2)
        if IS._dist_transform is not None:
            ax[2,0].imshow(IS._dist_transform)
        ax[2,1].imshow(im)
        im_name = "".join( image_path.split("/")[-1].split(".")[:-1] )
        plt.savefig(os.path.join(subdir,f"{im_name}_{naming_model}.png"))
        fig,ax = plt.subplots()
        ax.imshow(im)
        plt.savefig(os.path.join(subdir,f"{im_name}_color_{naming_model}.png"))
        ii += 1
        plt.close('all')

        df_experiment = pd.concat([df_experiment,df_image])
    df_experiment.to_csv(os.path.join(subdir,image_dir.split("/")[-1]+f"_ML_{naming_model}.csv"))

