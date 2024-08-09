# Goal of this script is to make several variations of 
# models for easier cross checking of performance
import sys
sys.path.append("..")
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier

from facet_ml.classification.model_training import *
import model_utils
## Define path to folder of csv files ##
data_super_folder = os.path.join("..","ProcessedData","Training_Data_20231106")

# 
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

with open("model_trials.json","r") as f:
    model_trials = json.load(f)

key_list = [#"C-MC_I-P",
            "C_MC", # Crystal vs Multiple Crystal
            "C-MC_I", # Crystalline vs non-crystal
            #"I_P",
            #"C_MC-I-P",
            "C_MC_I" # All
            ]
f1_dict = {}
for trial_name,param_dict in model_trials.items():
    for key in key_list:
        list_labels = df_total.Labels.unique()
        for label in list_labels:
            print(label,len(df_total[df_total.Labels == label]))

        if key == "C-MC_I-P" or key == "C-MC_I":
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

        # Train model
        my_trainer = ModelTrainer(df_run,
                    model_class=RandomForestClassifier,
                    labels=labels,
                    **param_dict)
        
        my_trainer.best_model_loop(50)

        # Save Model
        save_folder = "2023_"+trial_name
        save_directory = os.path.join("..","facet_ml","static","Models",save_folder)
        os.makedirs(save_directory,exist_ok=True)
        save_path = os.path.join(save_directory,f"RF_{key}.sav")
        model = my_trainer.best_run_dict["model"]

        with open(save_path,"wb") as f:
            pickle.dump(model,f)
        f1_dict[f"{trial_name} {key}"] = my_trainer.best_run_dict["f1_score"]


print(f1_dict)