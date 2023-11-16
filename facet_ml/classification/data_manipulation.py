from sklearn.preprocessing import OneHotEncoder
import numpy as np 
import pandas as pd

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

