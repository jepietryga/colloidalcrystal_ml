from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from pathlib import Path
import re


def adjust_df_crystal_noncrystal_data(df: pd.DataFrame):
    """
    Given a dataframe, change all of its labels to be either crystalline or non-crystalline
                  ALL DATA
        ------>   /      \
        Crystalline      Non-crystalline________
           /   \                   /            \
    Crystal  Multiple-Crystal  Incomplete     Poorly Segmented
    """
    df_copy = df.replace(["Multiple Crystal", "Crystal"], "Crystalline")
    df_copy = df_copy.replace(["Poorly Segmented", "Incomplete"], "Not Crystalline")

    df_copy.dropna(subset=["Labels"], inplace=True)
    return df_copy


def adjust_df_list_values(
    df: pd.DataFrame, label_list: list[str] = ["Crystal", "Multiple Crystal"]
):
    """
    Given a dataframe, keep only values in list (split the second level)
                  ALL DATA
                 /      \
        Crystalline      Non-crystalline_______
      ---> /   \                   /            \
    Crystal  Multiple-Crystal  Incomplete     Poorly Segmented
    """
    df_copy = df[df["Labels"].isin(label_list)]
    return df_copy


def create_formatted_df(csv_list, overwrite_string: str = None):
    """
    Given a list of csv files, create dataframes w/ file information
    If overwrite string is used, use it for EVERY csv in the overwrite string list
    """
    regex = "(?<=[_|\s])?([^_]+)-([^_]+)(?=[_|\s])?"

    df_arr = []
    for csv_path in csv_list:
        if overwrite_string is None:
            search_str = Path(csv_path).stem
        else:
            search_str = overwrite_string

        found = re.findall(regex, search_str)
        identifier_kwargs = {key: val for key, val in found}
        identifier_kwargs = identifier_kwargs | {
            "search_str": search_str,
            "path": csv_path,
        }
        # Update these values into every column of the dataframe
        df_temp = pd.read_csv(csv_path)
        for identifier, val in identifier_kwargs.items():
            df_temp[identifier] = [val] * len(df_temp)

        df_arr.append(df_temp)

    df_final = pd.concat(df_arr)
    return df_final
