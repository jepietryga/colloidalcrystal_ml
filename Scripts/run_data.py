import sys
sys.path.append("..")
import glob
import tqdm
from Utility import run_data_utils as rdu 

# Define a list of experiments where an experiemnt 
# is a FOLDER of images of similar variables (oven, mixing, etc.)

image_folder = "../Images"
experiment_list = glob.glob(image_folder+"/2023_02/*")

for experiment in tqdm.tqdm(experiment_list):
    rdu.run_experiment_folder(experiment)