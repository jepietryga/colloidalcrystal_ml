import sys
sys.path.append("..")
from facet_ml.classification import model_training as mt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import tqdm
ModelTrainer = mt.ModelTrainer

df_load = pd.read_csv("ProcessedData/Training_Data_20240216/2024_02_16_Rachel-C_Training.csv")
df_load = df_load.dropna(axis=0)
myTrainer = ModelTrainer(df_load,
             model_class=RandomForestClassifier,
             features="default_features",
             model_params="original",
             labels=["Crystal","Multiple Crystal","Incomplete","Poorly Segmented"]
             )

myTrainer.train_test_split()
myTrainer.fit()
myTrainer.predict()
myTrainer.score()
print(myTrainer.f1)

f1_list = []
tree_list = range(100)
for ii in tqdm.tqdm(tree_list):
    myTrainer.reset_best_run()
    myTrainer.model_params["n_estimators"] = ii+1
    myTrainer.best_model_loop(100)
    f1_list.append(myTrainer.best_run_dict["f1_score"])

plt.plot(tree_list,f1_list)
plt.savefig("tree.png")
print(myTrainer.best_run_dict)