import sys
sys.path.append("..")
import glob
import tqdm
from Utility import run_data_utils as rdu 

# Define a list of experiments where an experiemnt 
# is a FOLDER of images of similar variables (oven, mixing, etc.)

image_folder = "../Images"
input_folder = "Diagnostic_Images"
#input_folder =  "Organized images 20230613"
#input_folder = "Organized_Images"
file_id = "Testing"
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
        #rdu.run_experiment_folder_tile(experiment,results_folder=f"../Results/{input_folder}_Results_20230620",threshold_mode=th_mode,edge_modification=edge_mode)
        rdu.run_experiment_folder_tile(experiment,results_folder=f"../Results/{input_folder}_{file_id}",threshold_mode=th_mode,edge_modification=edge_mode)
