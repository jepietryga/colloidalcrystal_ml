import sys
sys.path.append("..")
from facet_ml.classification import model_training as mt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import tqdm
import os
import glob
import model_utils
import copy
import pickle
import numpy as np
ModelTrainer = mt.ModelTrainer

## Define path to folder of csv files ##
data_super_folder = os.path.join("..","ProcessedData","Training_Data_20231106")

## Define save folder
save_folder = "2023_11_models_length-agnostic"
save_directory = os.path.join("..","facet_ml","static","Models",save_folder)

### MAIN CODE ###

# Recreating these models w/ facet_score included from new labeling
# Need to first recombined everything as one dataframe

csv_paths = glob.glob(os.path.join(data_super_folder,"*"))
df_list = []
for path in csv_paths:
    df = pd.read_csv(path)
    df_list.append(df)
df_total = pd.concat(df_list)
df_total.replace([np.inf, -np.inf], np.nan, inplace=True)
df_total.dropna(axis=0,inplace=True)
df_total.to_csv("total.csv")
## Train each model:
# C-MC_I-P
# C_MC
# I_P
# C_MC-I-P
# C_MC_I_P
key_list = ["C-MC_I",
            "C_MC",
            #"I_P",
            #"C_MC-I-P",
            "C_MC_I"
            ]
for key in key_list:
    list_labels = df_total.Labels.unique()
    for label in list_labels:
        print(label,len(df_total[df_total.Labels == label]))

    if key == "C-MC_I":
        df_run = model_utils.adjust_df_crystal_noncrystal_data(df_total)
        labels = ["Crystalline","Not Crystalline"]
    if key == "C_MC":
        #df_run = model_utils.adjust_df_list_values(df,["Crystal","Multiple Crystal"])
        crystal_bool = df_total.Labels=="Crystal"
        multi_bool = df_total.Labels=="Multiple Crystal"
        df_run = df_total[crystal_bool | multi_bool]
        labels = ["Crystal","Multiple Crystal"]
        print(df_run.Labels.unique(),len(df_run))
    if key == "I_P":
        df_run = model_utils.adjust_df_list_values(df,["Incomplete","Poorly Segmented"])
        labels = ["Incomplete","Poorly Segmented"]
    if key == "C_MC-I-P":
        new_name = "Not Crystal"
        to_replace = ["Poorly Segmented","Multiple Crystal","Incomplete"]
        df_run = df_total.replace(to_replace,new_name)
        labels = ["Crystal","Not Crystal"]
    if key == "C_MC_I":
        df_run = copy.deepcopy(df_total)
        labels = ["Crystal","Multiple Crystal","Incomplete"]

    list_labels = df_run.Labels.unique()
    for label in list_labels:
        print(label,len(df_run[df_run.Labels == label]))


    my_trainer = ModelTrainer(df_run,
                 model_class=RandomForestClassifier,
                 model_params="original",
                 features="default_features-agnostic",
                 labels=labels)
    
    my_trainer.best_model_loop(50)

    # Save
    os.makedirs(save_directory,exist_ok=True)
    save_path = os.path.join(save_directory,f"RF_{key}.sav")
    
    with open(save_path,"wb") as f:
        print(f"{key}:{my_trainer.best_run_dict['f1_score']:.3f}")
        print(my_trainer.best_run_dict["model"].predict(my_trainer.best_run_dict["X_test"]))
        pickle.dump(my_trainer.best_run_dict,f)
