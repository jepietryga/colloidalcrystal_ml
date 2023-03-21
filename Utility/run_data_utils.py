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
    for label in ["Crystalline","Not Crystalline"]:
        
        df_temp = df[df["Labels"] == label]
        if len(df_temp) == 0:
            continue
        model = model_dict["C_MC"] if "Crystalline" else model_dict["I_P"]

        df_temp["Labels"] = model.predict(df_temp[features_list])

        df_subset_list.append(df_temp)
    
    df_img = pd.concat(df_subset_list)
    df_img.sort_values(by="Region")

    return df_img

def run_experiment_folder(experiment_path:str,
    model_mode:str="multitree",
    model_folder:str="../Models/model_set__original_2023_classifier",
    config_path:str="../Utility/config.json",
    feature_str:str="default_features",
    results_folder:str="../Results"
    ):
    '''
    Run an experiment (A folder full of images or folders of images) through a model_mode
    Get results from this experiment and save them in ../Results based on experiment folder name
    '''

    # Get all image paths and those in deeper subfolders
    img_paths = glob.glob(os.path.join(experiment_path,"**/*.tif"),recursive=True)
    
    df_experiment = pd.DataFrame()
    for img in img_paths:
        # Get dataframe and features
        IS = ImageSegmenter(img,threshold_mode="ensemble",edge_modification=True)
        df_img = IS.df
        features_list = model_utils.load_features(config_path,feature_str)

        # Numerical errors (divide by 0)
        df_img.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        df_img.dropna(subset=features_list,inplace=True)

        # Deploy model workflow
        if model_mode == "multitree":
            df_img = multitree_model(df_img,features_list,model_folder)
        else:
            raise Exception(f"{model_mode} not supported")
        
        df_experiment = pd.concat([df_experiment, df_img])

    save_name = experiment_path.split("/")[-1]+".csv"
    save_path = os.path.join(results_folder,save_name)
    df_experiment.to_csv(save_path)



