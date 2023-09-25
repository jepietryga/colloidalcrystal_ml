from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import glob
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from Utility import model_utils
from Utility.segmentation_utils import ImageSegmenter

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
                label_arr.append("Crystalline")
            if index == 1:
                label_arr.append("Not Crystalline")
                
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

def multitree_model(df,features_list,model_folder):
    '''
    Given a dataframe of features and a folder of models trained on such features, determine labels
    '''
    # Determine and Load Models
    model_id_list = [
        "C-MC_I-P", # NOTE: Returned labels are "Crystalline" and "Not Crystalline"
        "C_MC",
        "I_P",
        ]
    model_dict = {}
    for model_id in model_id_list:
        model_path = os.path.join(model_folder,f"{model_id}.sav")
        with open(model_path,"rb") as f:
            model = pickle.load(f)["model"]
        model_dict[model_id] = model

    # Begin Predictions
    X = df[features_list]
    labels = model_dict["C-MC_I-P"].predict(X)
    df["Labels"] = labels

    # Crystalline or Not Crystalline split
    df_subset_list = []
    print(df.Labels.unique())
    for label in ["Crystalline","Not Crystalline"]:
        
        df_temp = df[df["Labels"] == label]
        if len(df_temp) == 0:
            print(f"WARNING: {label} has no numbers")
            continue
        model = model_dict["C_MC"] if label == "Crystalline" else model_dict["I_P"]

        df_temp["Labels"] = model.predict(df_temp[features_list])
        print(f"For label {label}, get {np.unique(df_temp['Labels'])}")
        df_subset_list.append(df_temp)
    
    df_img = pd.concat(df_subset_list)
    df_img.sort_values(by="Region")

    return df_img

def run_experiment_folder(experiment_path:str,
    model_mode:str="multitree",
    model_folder:str="../Models/model_set__original_2023_classifier",
    config_path:str="../Utility/config.json",
    feature_str:str="default_features",
    results_folder:str="../Results",
    threshold_mode:str="ensemble",
    edge_modification=None,
    ):
    '''
    Run an experiment (A folder full of images or folders of images) through a model_mode
    Get results from this experiment and save them in ../Results based on experiment folder name
    '''
    added_tags = []

    # define additional tags and save paths
    edge_str = edge_modification
    if not isinstance(edge_str,str):
        edge_str = "None"
    print(f"DEBUG: {experiment_path}")
    img_ext = ["tif","png","jpg","bmp"]
    experiment_id = experiment_path.split("/")[-1]
    added_tags.append(experiment_id)
    added_tags.append(f"edge-{edge_str}")
    added_tags.append(f"thresh-{threshold_mode}")
    extended_experiment_id = "_".join(added_tags)
    save_name = extended_experiment_id+".csv"
    save_path = os.path.join(results_folder,save_name)
    
    # Get all image paths and those in deeper subfolders
    img_paths = glob.glob(os.path.join(experiment_path,"**/*"),recursive=True)
    img_paths = [p for p in img_paths if any( ext in p.split(".")[-1] for ext in img_ext)]
    df_experiment = pd.DataFrame()
    for img in img_paths:
        # Get dataframe and features
        IS = ImageSegmenter(img,threshold_mode=threshold_mode,edge_modification=edge_modification)
        df_img = IS.df
        features_list = model_utils.load_features(config_path,feature_str)

        # Numerical errors (divide by 0)
        print(f"Pre-Proc: {img} has len {len(df_img)})")
        df_img.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        df_img.dropna(subset=features_list,inplace=True)

        print(f"Post-Proc: {img} has len {len(df_img)})")

        # Deploy model workflow
        if model_mode == "multitree":
            df_img = multitree_model(df_img,features_list,model_folder)
        else:
            raise Exception(f"{model_mode} not supported")

        # checking images
        cimg_id = "".join( img.split("/")[-1].split(".")[:-1] )
        cimg_dir = os.path.join(results_folder,extended_experiment_id+"_ColorImages")
        cimg_path = os.path.join(cimg_dir,cimg_id+"_Color.png")
        mimg_path = os.path.join(cimg_dir,cimg_id+"_Markers.png")
        
        os.makedirs(cimg_dir,exist_ok=True)
        c_img = model_utils.generate_region_colored_image(df_img,IS)
        plt.imsave(cimg_path,c_img)
        plt.imsave(mimg_path,IS.markers2)
        if edge_modification:
            eimg_path = os.path.join(cimg_dir,cimg_id+"_Edges.png")
            plt.imsave(eimg_path,IS._edge_highlight)

        df_experiment = pd.concat([df_experiment, df_img])

    df_experiment.to_csv(save_path)

def run_experiment_folder_tile(experiment_path:str,
    model_mode:str="multitree",
    model_folder:str="../Models/model_set__original_2023_classifier",
    config_path:str="../Utility/config.json",
    feature_str:str="default_features",
    results_folder:str="../Results",
    threshold_mode:str="ensemble",
    edge_modification=None,
    ):
    '''
    Run an experiment (A folder full of images or folders of images) through a model_mode
    Get results from this experiment and save them in ../Results based on experiment folder name
    6_set describes the output visualization
    '''
    # Make sure folder is there for results
    os.makedirs(results_folder,exist_ok=True)
    added_tags = []

    # define additional tags and save paths
    edge_str = edge_modification
    if not isinstance(edge_str,str):
        edge_str = "None"
    print(f"DEBUG: {experiment_path}")
    img_ext = ["tif","png","jpg","bmp"]
    experiment_id = experiment_path.split("/")[-1]
    added_tags.append(experiment_id)
    added_tags.append(f"edge-{edge_str}")
    added_tags.append(f"thresh-{threshold_mode}")
    extended_experiment_id = "_".join(added_tags)
    save_name = extended_experiment_id+".csv"
    save_path = os.path.join(results_folder,save_name)
    
    # Get all image paths and those in deeper subfolders
    img_paths = glob.glob(os.path.join(experiment_path,"**/*"),recursive=True)
    img_paths = [p for p in img_paths if any( ext in p.split(".")[-1] for ext in img_ext)]
    df_experiment = pd.DataFrame()
    for img in img_paths:
        # Get dataframe and features
        IS = ImageSegmenter(img,threshold_mode=threshold_mode,edge_modification=edge_modification)
        df_img = IS.df
        features_list = model_utils.load_features(config_path,feature_str)

        # Numerical errors (divide by 0)
        print(f"Pre-Proc: {img} has len {len(df_img)})")
        df_img.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        df_img.dropna(subset=features_list,inplace=True)

        print(f"Post-Proc: {img} has len {len(df_img)})")

        # Deploy model workflow
        if model_mode == "multitree":
            df_img = multitree_model(df_img,features_list,model_folder)
        else:
            raise Exception(f"{model_mode} not supported")

        # checking images
        # edges thresh
        # markers markers 2
        # dist_transform  colored_image
        fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(12,18))

        if edge_modification:
            ax[0,0].imshow(IS._edge_highlight); ax[0,0].set_title(f"Edge Highlight: {edge_modification} {IS._edge_stats}")
        else:
            ax[0,0].imshow(IS.img2); ax[0,0].set_title(f"Edge Highlight: {edge_modification}")
        ax[0,1].imshow(IS.thresh); ax[0,1].set_title(f"Thresh {threshold_mode}")
        ax[1,0].imshow(IS.markers); ax[1,0].set_title("Markers")
        ax[1,1].imshow(IS.markers2); ax[1,1].set_title("Markers2")
        ax[2,0].imshow(IS._dist_transform); ax[2,0].set_title("Dist. Transform")
        ax[2,1].imshow(model_utils.generate_region_colored_image(df_img,IS)); ax[2,1].set_title("Colored Image")
        for ax_oi in ax.ravel():
            print(ax_oi)
            ax_oi.set_xticks([])
            ax_oi.set_yticks([])

        fig.subplots_adjust(hspace=0,wspace=0)
        
        cimg_id = "".join( img.split("/")[-1].split(".")[:-1] )
        cimg_dir = os.path.join(results_folder,extended_experiment_id+"_img-tile")
        cimg_path = os.path.join(cimg_dir,cimg_id+"_img-tile.png")
        
        os.makedirs(cimg_dir,exist_ok=True)
        #c_img = model_utils.generate_region_colored_image(df_img,IS)
        #plt.imsave(cimg_path,c_img)
        fig.savefig(cimg_path)
        plt.close(fig)
        df_experiment = pd.concat([df_experiment, df_img])

    df_experiment.to_csv(save_path)





