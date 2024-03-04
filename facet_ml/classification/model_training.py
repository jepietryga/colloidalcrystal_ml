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
from typing import Union

from pathlib import Path

CONFIG_PATH = os.path.join(Path(__file__).parent,"config_files")

def load_feature_config(key:str) -> list:
    '''
    From the config_features.json file, return a tuple list of feature names
    '''
    config_path = os.path.join(CONFIG_PATH,"config_features.json")
    with open(config_path,"r") as f:
        config_dict = json.load(f) 
    return config_dict[key]

def load_model_config(key:str):
    config_path = os.path.join(CONFIG_PATH,"config_model.json")
    with open(config_path,"r") as f:
        config_dict = json.load(f) 
    return config_dict[key]

class ModelTrainer():

    def __init__(self,
                df:pd.DataFrame,
                model_class:callable,
                model_params:Union[str,dict],
                features:Union[str,list],
                targets:list[str]="Labels",
                labels:list[str]=["Crystalline","Not Crystalline"],
                test_size:float=.2,
                seed:int=None,
                regressor:bool=False
                ):
        '''
        ModelTrainer class wraps a lot of model training functionality together for easy tracking
        Can be done outside class if generally familiar w/ pipeline

        Args:
            df (pd.DataFrame) : Main dataframe the model will be considering
            model_class (Callable) : A class that has "fit" and "predict" methods 
            model_params (str,dict) : If str, try and load from the config_model. 
                                    If dict, these are treated as kwargs for the model_class init
            features (str,dict) : If str, try and load from the config_features
                                    If list, use these features directly
        '''
        self.df = df
        self.model_class = model_class
        self.model_params = model_params if isinstance(model_params,dict) else load_model_config(model_params)
        self.features = features if isinstance(features,list) else load_feature_config(features)
        self.targets = targets
        self.labels = labels
        self.seed = seed
        self.regressor = regressor
        self.test_size = test_size

        # Model variables
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Find Best Model variables
        self.best_run_dict = {
            "model":None,
            "f1_score":-np.inf,
            "confusion_matrix":None,
            "X_test":None,
            "X_train":None,
            "y_test":None,
            "y_train":None,
        }
        self.logging_f1 = None

    def train_test_split(self,
                         test_size:float=None,
                         random_state:int=None):
        '''
        Split the training, test data
        '''
        if not test_size:
            test_size=self.test_size

        self.X = self.df[self.features]
        self.y = self.df[self.targets]

        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,
                                                                             test_size=test_size,
                                                                             random_state=random_state,
                                                                             stratify=self.y)

        return self.X_train,self.X_test,self.y_train,self.y_test

    def fit(self):
        self.model = self.model_class(**self.model_params)
        self.model.fit(self.X_train,self.y_train)

    def predict(self,
                X=None):
        if not X:
            X = self.X_test
        self.y_pred = self.model.predict(X)
        return

    def score(self):
        self.f1 = metrics.f1_score(self.y_test,self.y_pred,average='macro')
        
        self.confusion_matrix = metrics.multilabel_confusion_matrix(self.y_test,self.y_pred,
            labels=self.labels)
        
    def update_best_run(self):
        self.score()
        if self.f1 > self.best_run_dict["f1_score"]:
                self.best_run_dict = {
                    "model":copy.deepcopy(self.model),
                    "f1_score":copy.deepcopy(self.f1),
                    "confusion_matrix":copy.deepcopy(self.confusion_matrix),
                    "X_test":copy.deepcopy(self.X_test),
                    "X_train":copy.deepcopy(self.X_train),
                    "y_test":copy.deepcopy(self.y_test),
                    "y_train":copy.deepcopy(self.y_train),
                }

    def reset_best_run(self):
        self.best_run_dict = {
            "model":None,
            "f1_score":-np.inf,
            "confusion_matrix":None,
            "X_test":None,
            "X_train":None,
            "y_test":None,
            "y_train":None,
        }
    
    def best_model_loop(self,
                        iterations:int=100):
        '''
        Loop over random instantiations of the model for a set number of iterations
        This can help with RandomForestClassifiers to get a reasonable first guess

        Args:
            iterations (int):Numer of times to run the fit-predict-score loop

        '''
        logging_f1 = []
        for seed in tqdm.tqdm(np.arange(iterations)):
            self.train_test_split(random_state=seed)
            self.fit()
            self.predict()
            self.score()

            self.update_best_run()

            # Log data
            logging_f1.append(self.f1)

        return self.logging_f1
            

def replace_and_clean_labels_df(df:pd.DataFrame,replace_list:list,) -> pd.DataFrame:
    '''
    Given a dataframe, use the replace list (populated with ( [terms to replace], replace) ),
    replace each value in the Labels column
    Then, clear any other held value
    '''
    df_copy = df.copy()
    label_list = []
    for targets, replacer in replace_list:
        label_list.append(replacer)
        df_copy.replace(targets,replacer,inplace=True)

    df_copy = df_copy[df_copy["Labels"].isin(label_list)]

    df_copy.dropna(subset=['Labels'],inplace=True)
    return df_copy
