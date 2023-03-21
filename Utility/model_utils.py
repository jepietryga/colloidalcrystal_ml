# Series of helper functions for doing model training
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats as stat
import numpy as np 
#import forestsci
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
import json

import sys
sys.path.append("..")

CONFIG_JSON = "config.json"

def load_data(path_to_training):
    '''
    Load the trianing data (which comes with geometric features)
    '''
    df = pd.read_csv(path_to_training)
    return df

def load_params(param_str:str="original",config_path=CONFIG_JSON):
    '''
    Return dict (Kwargs) for running model of choice
    '''

    with open(config_path,"r") as f:
        config_dict = json.load(f)
    
    return config_dict["model_params"][param_str]

def load_features(config_path,feature_str:str="default_features"):
    
    with open(config_path,"r") as f:
        config_dict = json.load(f)
        features_list = config_dict[feature_str]
    return features_list

def adjust_df_crystal_noncrystal_data(df:pd.DataFrame):
    '''
    Given a dataframe, change all of its labels to be either crystalline or non-crystalline
                  ALL DATA
        ------>   /      \
        Crystalline      Non-crystalline________
           /   \                   /            \
    Crystal  Multiple-Crystal  Incomplete     Poorly Segmented
    '''
    df_copy = df.replace(['Multiple Crystal','Crystal'],'Crystalline')
    df_copy = df_copy.replace(['Poorly Segmented','Incomplete'],"Not Crystalline")

    #print(type(df_copy))
    #print(df_copy)
    df_copy.dropna(subset=['Labels'],inplace=True)
    return df_copy

def adjust_df_list_values(df:pd.DataFrame,label_list:list[str]=["Crystal","Multiple Crystal"]):
    '''
    Given a dataframe, keep only values in list (split the second level)
                  ALL DATA
                 /      \
        Crystalline      Non-crystalline_______
      ---> /   \                   /            \
    Crystal  Multiple-Crystal  Incomplete     Poorly Segmented
    '''
    df_copy = df[df["Labels"].isin(label_list)]
    return df_copy

def split_feature_labels(df:pd.DataFrame,features_list:list[str] = None,
                         targets_list:list[str] = ["Labels"],
                         config_path=CONFIG_JSON,
                         regressor=True):
    
    if features_list is None:
        features_list = load_features(config_path)
    
    ohe = OneHotEncoder(sparse=False)

    X = df[features_list]
    y = ohe.fit_transform(df[targets_list]) if regressor else df[targets_list]

    return X,y,ohe
    

def categorical_data_translator(passed_list):
    '''
    This is hard-coded since we know our own classifications
    '''

    num_list = []
    index_track = 0
    for item in passed_list:
        if item == 'Crystal':
            num_list.append(2)
        elif item == 'Multiple Crystal':
            num_list.append(3)
        elif item == 'Poorly Segmented':
            num_list.append(1)
        elif item == 'Incomplete':
            num_list.append(0)
        else:
            print(f'Error: {item} is unknown ID at {index_track}')
            exit()
        index_track = index_track + 1
    
    return num_list


def success_of_guess(y_pred,y_test,ohe):
    '''
    Given the predicted results, for each label, how does our model perform?

    Args:
    y_pred (array) : Our predicted values in OneHotEncoding
    y_test (array) : Our test values in OneHotEncoding
    ohe (OneHotEncoder) : Our OneHotEncoder (for translating meaning)

    Returns:
    success (ndarray) : Successful guesses for each label
    failed_to_guess (ndarry) : Number of a times a label should've been guessed, but was missed
    incorrectly_guessed (ndarry) : Number of times a label was mistakenly guessed
    paired_guess (ndarry) : 2D Array where -1 indicates the guess and 1 indicates the correct answer. 
                            All 0s implies a successful guess
    '''
    success = np.zeros([np.size(y_pred[0]),])
    failed_to_guess = np.zeros([np.size(y_pred[0]),])
    incorrectly_guessed = np.zeros([np.size(y_pred[0]),])
    paired_guess = y_test-y_pred
    for ii in np.arange(np.shape(y_pred)[0]):
        if np.where(y_pred[ii] == 1) == np.where(y_test[ii] == 1):
            success += y_pred[ii]
        else:
            incorrectly_guessed += y_pred[ii]
            failed_to_guess += y_test[ii]
    
    labels_list = ohe.get_feature_names_out()

    for ii in np.arange(np.size(labels_list)):
        recall = success[ii]/(success[ii]+failed_to_guess[ii])
        precision = success[ii]/(success[ii]+incorrectly_guessed[ii])
        f1 = 2*precision*recall/(precision+recall)
        print(f'{labels_list[ii]} -> Precision = {precision}, Recall = {recall}, F1 = {f1}')
    accuracy = np.sum(success)/(np.shape(y_pred)[0])
    print(f'Run Accuracy : {accuracy}')