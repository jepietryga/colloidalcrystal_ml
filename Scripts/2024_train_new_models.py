# Goal of this script is to make several variations of
# models for easier cross checking of performance
import sys

sys.path.append("..")
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier

from facet_ml.classification.model_training import *

# import model_utils

# Save Naming f"{save_id}_{trial_name}"  <-- Later on
save_id = "2024_02_"
model_directory = os.path.join("..", "facet_ml", "static", "Models")


# Input Data
processed_data = Path(__file__).parent.parent / "ProcessedData"
csv_path = (
    processed_data / "Training_Data_20240216" / "2024_02_27_Rachel-C_Processed.csv"
)
df_total = pd.read_csv(csv_path)

# Input model_parameters
param_path = Path(__file__).parent / "model_configs_2024.json"

### MAIN CODE ###

# Recreating these models w/ facet_score included from new labeling

# Need to first recombined everything as one dataframe
# Clean Data
df_total = df_total.loc[:, ~df_total.columns.str.contains("^Unnamed")]
df_total.replace([np.inf, -np.inf], np.nan, inplace=True)
df_total.dropna(axis=0, inplace=True)

with open(param_path, "r") as f:
    model_trials = json.load(f)


key_list = [  # "C-MC_I-P",
    "C_MC",  # Crystal vs Multiple Crystal
    "C-MC_I",  # Crystalline vs non-crystal
    # "I_P",
    # "C_MC-I-P",
    "C_MC_I",  # All
]
f1_dict = {}

for trial_name, param_dict in model_trials.items():
    for key in key_list:
        list_labels = df_total.Labels.unique()

        if key == "C-MC_I-P" or key == "C-MC_I":
            replace_list = [
                (["C", "MC"], "Crystalline"),
                (["I", "P"], "Not Crystalline"),
            ]
            labels = ["Crystalline", "Not Crystalline"]

        if key == "C_MC":
            replace_list = [("C", "Crystal"), ("MC", "Multiple Crystal")]
            labels = ["Crystal", "Multiple Crystal"]

        if key == "I_P":
            replace_list = [("I", "Incomplete"), ("PS", "Poorly Segmented")]
            labels = ["Incomplete", "Poorly Segmented"]

        if key == "C_MC-I-P":
            replace_list = [("C", "Crystal"), (["MC", "I", "PS"], "Not Crystal")]
            labels = ["Crystal", "Not Crystal"]

        if key == "C_MC_I":
            replace_list = [
                ("C", "Crystal"),
                ("MC", "Multiple Crystal"),
                ("I", "Incomplete"),
            ]

            labels = ["Crystal", "Multiple Crystal", "Incomplete"]

        # Replace and clean data
        df_run = replace_and_clean_labels_df(df_total, replace_list)
        print(df_run.Labels.value_counts())

        # Train model
        my_trainer = ModelTrainer(
            df_run, model_class=RandomForestClassifier, labels=labels, **param_dict
        )

        my_trainer.best_model_loop(100)

        # Save Model
        save_folder = f"{save_id}_{trial_name}"
        save_directory = os.path.join(model_directory, save_folder)
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, f"RF_{key}.sav")
        model = my_trainer.best_run_dict

        print(save_path)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        f1_dict[f"{trial_name} {key}"] = my_trainer.best_run_dict["f1_score"]


print(f1_dict)
