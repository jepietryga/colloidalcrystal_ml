# Purpose of this code is to take in new data and process it

import sys
sys.path.append("..")
import pickle
import os
from sklearn import RandomForestRegressor 

from Utility.segmentation_utils import *

# Define folders
model_folder = "../Models"
image_folder = "../Images"
result_folder = "../Results"
# Load models
model_names = ["RF_C-MC_I-P.sav","RF_C_MC.sav","RF_I_P.sav"]
rf_CMC_IP, rf_C_MC, rf_I_P = [pickle.load(open(os.path.join(model_folder,model), 'rb'))\
                              for model in model_names]

# Grab image folder
super_folder = glob.glob("../Images/Organized/*/")
image_dir = super_folder[0]

## Grab image
### IS the Image
### get df of Image
### 