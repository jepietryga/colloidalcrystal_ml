import sys
sys.path.append("..")
import pandas as pd
from Utility import process_data_utils as pdu
import glob
import numpy as np
import os
from pathlib import Path

# Load in csvs
target_method = "ensemble"
#path = "/Users/jacobpietryga/Desktop/Academics/colloidal_crystal_ML/Results/L-1_nM-10_mixing-F_rate_fast_ensemble/L-1_nM-10_mixing-F_rate_fast_ML_sam.csv"
folder = f"../Results/2023_11_Organized_Images_{target_method}"
experiment_folders = [p for p in glob.glob(os.path.join(folder,"*")) if os.path.isdir(p)]

df_arr = []
for experiment in experiment_folders:
    
    _,folder_name = os.path.split(experiment)#Path(experiment).stem
    print(folder_name)
    csv_paths = glob.glob(os.path.join(experiment,"*.csv"))
    #df_folder = pdu.create_formatted_df(csv_paths,overwrite_string=folder_name)
    #df_arr.append(df_folder)
exit()
df_total = pd.concat(df_arr)

# Rearrange columns to be easier to read
column_list = df_total.columns.tolist()
col_index = column_list.index("Region")
df_total = df_total[column_list[col_index+1:]+column_list[:col_index+1]]

df_total.to_csv(f"../ProcessedData/2023_11_Organized_Images_{target_method}_total.csv")