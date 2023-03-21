from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os 
import sys
sys.path.append('..')
import json
import pickle
from sklearn.inspection import permutation_importance
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

import Utility.model_utils as model_utils

pd.options.mode.chained_assignment = None

# Define Save Paths
id =  "2023_classifier"
model_params = "original" # Inside config.json
model_folder = "../Models"
save_folder = os.path.join(model_folder,f"model_set__{model_params}_{id}")
os.makedirs(save_folder,exist_ok=True)

# Load Data
config_path="../Utility/config.json"
features = json.load(open(config_path,"r"))["default_features"]
df = model_utils.load_data("../Results/training_data.csv")

param_dict = model_utils.load_params(model_params,config_path)

#model_func = RandomForestRegressor; regressor_bool = True
model_func = RandomForestClassifier; regressor_bool = False

# ---- Main code ---- #
best_model_dict = {
    "C-MC_I-P":{
        "model":None,
        "f1_score":0,
    },
    "C_MC":{
        "model":None,
        "f1_score":0,
    },
    "I_P":{
        "model":None,
        "f1_score":0,
    },
    "C_MC-I_P":{
        "model":None,
        "f1_score":0,
    },
    "C_MC_I_P":{
        "model":None,
        "f1_score":0,
    },
}

# Crystalline vs. Not Crystalline Split
for key,_ in tqdm.tqdm(best_model_dict.items()):
    df = model_utils.load_data("../Results/training_data.csv")
    if key == "C-MC_I-P":
        df_sub = model_utils.adjust_df_crystal_noncrystal_data(df)
        labels = ["Crystalline","Not Crystalline"]
    if key == "C_MC":
        df_sub = model_utils.adjust_df_list_values(df,["Crystal","Multiple Crystal"])
        labels = ["Crystal","Multiple Crystal"]
    if key == "I_P":
        df_sub = model_utils.adjust_df_list_values(df,["Incomplete","Poorly Segmented"])
        labels = ["Incomplete","Poorly Segmented"]
    if key == "C_MC-I_P":
        new_name = "Not Crystal"
        to_replace = ["Poorly Segmented","Multiple Crystal","Incomplete"]
        df_sub = df.replace(to_replace,new_name)
        labels = ["Crystal","Not Crystal"]
    if key == "C_MC_I_P":
        df_sub = copy.deepcopy(df)
        labels = ["Crystal","Multiple Crystal","Incomplete","Poorly Segmented"]

    df_sub.dropna(inplace=True)
    for seed in tqdm.tqdm(np.arange(10)):
        X,y,ohe = model_utils.split_feature_labels(df_sub,config_path=config_path,features_list=features,
                    regressor=regressor_bool)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=seed)
        y_train = y_train if regressor_bool else np.ravel(y_train) # mutes errors
        model = model_func(**param_dict)
        model.fit(X_train,y_train)

        # Check accuracy, update best model
        y_pred = model.predict(X_test)
        # If regressor, get the maximum probability item
        if regressor_bool:
            y_pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
        #y_pred = np.round(y_pred) if regressor_bool else y_pred

        accuracy = metrics.f1_score(y_test,y_pred,average='macro')

        
        confusion_matrix = metrics.multilabel_confusion_matrix(y_test,y_pred,
            labels=labels)
        if accuracy > best_model_dict[key]["f1_score"]:
            run_dict = {
                "model":model,
                "f1_score":accuracy,
                "confusion_matrix":list(zip(labels,confusion_matrix))
            }
            best_model_dict[key] = run_dict

    # Save model
    save_name = f"{key}.sav"
    save_path = os.path.join(save_folder,save_name)
    pickle.dump(best_model_dict[key], open(save_path,"wb"))

    # Feature Importance analysis
    result = permutation_importance(best_model_dict[key]["model"], X_test, y_test, n_repeats=20, random_state=seed, n_jobs=4)
    forest_importances = pd.Series(result.importances_mean, index=features)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    
    save_name = f"FI_{key}.png"
    save_path = os.path.join(save_folder,save_name)
    plt.savefig(save_path)
    
# Print Summaries

for key,item in best_model_dict.items():
    val = item["f1_score"]
    print(f"{key} F1 Score: {val}")
    for ii,matrix in enumerate(item["confusion_matrix"]):
        print(matrix[0])
        print(matrix[1])





