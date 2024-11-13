from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
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
                n_splits:int = 5,
                seed:int=None,
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
            targets (list[str]) : List of training targets
            labels (list[str]) : In the Label's target, only target these labels and remove others
            test_size (float) : Size of the RandomForest test set
            seed (int) : Seed for randomizer
        '''
        self.df = df
        self.model_class = model_class
        self.model_params = model_params if isinstance(model_params,dict) else load_model_config(model_params)
        self.features = features if isinstance(features,list) else load_feature_config(features)
        self.targets = targets
        self.labels = labels
        self.seed = seed
        self.n_splits = n_splits

        # Model variables
        self.model = None
        self.X = None
        self.y = None

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

    def _get_features_targets(self):
        '''
        Helper function to get X and y
        '''
        # Filtering
        self.df = self.df[self.df[self.targets].isin(self.labels)]
        self.df = self.df.dropna(subset=self.features + [self.targets])
        self.df = self.df[~self.df[self.features + [self.targets]].isin([np.inf, -np.inf]).any(axis=1)]  # Drop rows with inf

        self.X = self.df[self.features]
        self.y = self.df[self.targets]

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

    def fit(self, X_train, y_train):
        self.model = self.model_class(**self.model_params)
        self.model.fit(X_train,y_train)

    def predict(self, X):
        return self.model.predict(X)

    def score(self,y_test,y_pred):
        f1 = metrics.f1_score(y_test,y_pred,average='macro')
        
        confusion_matrix = metrics.multilabel_confusion_matrix(y_test,y_pred,
            labels=self.labels)
        
        return f1, confusion_matrix
        
    def update_best_run(self,
                        run_kwargs
                        ):
        if run_kwargs["f1_score"] > self.best_run_dict["f1_score"]:
                self.best_run_dict = run_kwargs

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
            iterations (int):Number of times to run the fit-predict-score loop

        '''
        self._get_features_targets()
        logging_f1 = []
        for seed in tqdm.tqdm(np.arange(iterations)):
            fold_f1_scores = []
            
            # Create a new instance of StratifiedKFold for each iteration with a random seed
            folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)

            for train_index,test_index in folds.split(self.X,self.y):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

                # self.train_test_split(random_state=seed)
                self.fit(X_train,y_train)
                y_pred = self.predict(X_test)
                f1, confusion_matrix = self.score(y_test,y_pred)

                run_info = {
                    "model":copy.deepcopy(self.model),
                    "f1_score":f1,
                    "confusion_matrix":confusion_matrix,
                    "X_test":X_test,
                    "X_train":X_train,
                    "y_test":y_test,
                    "y_train":y_train
                }

                fold_f1_scores.append(f1)
                self.update_best_run(run_kwargs=run_info)

            # Log data
            logging_f1.append(np.mean(fold_f1_scores))

        return logging_f1
            

def replace_and_clean_labels_df(df:pd.DataFrame,replace:list,) -> pd.DataFrame:
    '''
    Given a dataframe, perform replace operation and remove labels NOT associated with any new names

    Args:
        df (pd.DataFrame) : Dataframe to replace names in
        replace (dict) : Replacement dict to use for replacement
    '''
    df_copy = df.copy()
    label_list = []
    for targets, replacer in replace.items():
        label_list.append(replacer)
        df_copy.replace(targets,replacer,inplace=True)

    df_copy = df_copy[df_copy["Labels"].isin(label_list)]

    df_copy.dropna(subset=['Labels'],inplace=True)
    return df_copy
