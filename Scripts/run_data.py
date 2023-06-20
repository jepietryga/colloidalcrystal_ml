import sys
sys.path.append("..")
import glob
import tqdm
from Utility import run_data_utils as rdu 

# Define a list of experiments where an experiemnt 
# is a FOLDER of images of similar variables (oven, mixing, etc.)

image_folder = "../Images"
input_folder = "Diagnostic_Images"
experiment_list = glob.glob(image_folder+f"/{input_folder}/*")

for th_mode, edge_mode in [
["otsu",None],
#["local",None],
#["pixel",None],
["ensemble","darkbright"],
#["ensemble","variance"],
["ensemble",None],
#["ensemble","canny"]
]:
    for experiment in tqdm.tqdm(experiment_list):
        #rdu.run_experiment_folder(experiment,threshold_mode=th_mode,edge_modification=edge_mode)
        rdu.run_experiment_folder_tile(experiment,results_folder=f"../Results/{input_folder}_Results",threshold_mode=th_mode,edge_modification=edge_mode)
