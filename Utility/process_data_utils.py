import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.optimize import curve_fit
import scipy.stats as ss
import copy
import os
import matplotlib.pyplot as plt
import glob
import re
from functools import partial
from sklearn.metrics import r2_score
from pathlib import Path

def schultz_zimm(x,a,b):
    '''
    Fitting function that generalizes to macromolecular assembly (a=1,b=1 is Flory-Schulz for polymers)
    '''
    mu = np.mean(x)
    numerator = (a**(b+1))*np.exp(-a*x/mu)*x**(b)
    denominator = gamma(b+1)*mu
    p = numerator/denominator
    return p

def gaussian(x,A,mu,sig):
    '''
    Fitting function for Gaussian
    '''
    return A*np.exp(-(x-mu)**2/(2*sig)**2)

def get_frequency(df,histogram_bins,feature):
    '''
    Get the frequyency for each bin, explicitly
    Abstracted Implementation
    currently doesn't handle negative numbers
    '''
    freq = []
    lb = 0
    for h_bin in histogram_bins:
        subset = df[(df[feature] < h_bin) & (df[feature] > lb)]
        freq.append(len(subset))
        lb = h_bin
    return freq

def load_dataframe(file,directory_path:str="../Results"):
    '''
    Load dataframe with cleaner path control
    '''
    path_oi = os.path.join(directory_path,file)
    return pd.read_csv(path_oi)

def plot_feature_with_experiments_schultz_zimm(
        df,
        feature:str='area',
        experiment_list:list[str]=[],
        color_list:list[str]=["b","g","m","r","y"],
        labels = ["Crystal","Multiple Crystal","Incomplete","Poorly Segmented"]
    ):
    '''
    Given a feature found in ML Result csvs, plot it for each experiment listed

    '''
    # Load the dataframe dict for easy access
    df_dict = {}
    for experiment in experiment_list:
        df_sub = df[df.search_str == experiment]
        df_dict[experiment] = df_sub
    # Clipping Variable
    max_out = 2.50*10**6 if feature == 'area' else 2000

    # Set up figure and axis layout
    fig,ax = plt.subplots(len(labels),1,figsize=(15,15) )
    fig.tight_layout(pad=5.0)
    fig.suptitle(f"Distributions of {feature}")
    histogram_min = min([df[feature].min() for _,df in df_dict.items()])
    histogram_max = max([df[feature].max() for _,df in df_dict.items()])
    histogram_bins = np.round(np.linspace(histogram_min, max_out, 100))

    for ii,(name,df) in enumerate(df_dict.items()):
        c_oi=color_list[ii]
        total_counts = len(df)
        for jj,label in enumerate(labels):
            ax_oi=ax[jj]
            df_oi = df[df["Labels"] == label]
            df_oi = df_oi.replace([np.inf,-np.inf], np.nan)
            df_oi = df_oi[df_oi[feature].notna()]
            counts = len(df_oi)
            if counts == 0:
                continue

            # Schulz Plotting
            print(label)
            print(histogram_bins[0],len(df_oi))
            freq = get_frequency(df_oi,histogram_bins,feature)
            scale_factor = counts*histogram_bins[0]
            prob = np.array(freq)/(scale_factor)
            x_bins = histogram_bins + (histogram_bins[2]-histogram_bins[1])/2
            popt,pcov = curve_fit(schultz_zimm,x_bins,prob)
            a = popt[0]
            b = popt[1]
            molecular_weight = b/a
            molecular_number = (b+1)/a
            pdi = molecular_weight/molecular_number

            ax_oi.plot(x_bins,schultz_zimm(histogram_bins,*popt)*scale_factor,color=c_oi,
                label=f"{name} ({counts}/{total_counts} a: {a:.3f}, b: {b:.3f}, PDI: {pdi:.3f}")
            ax_oi.hist(x=df_oi[feature],bins=histogram_bins,color=c_oi,alpha=.5)
            ax_oi.legend()
            ax_oi.set_title(label)
    return fig

def row_fit_schultz_zimm(row:pd.Series,label=None,feature=None):
    '''
    Given a row from the formatted df(see next function), get the Schultz-Zimm a,b parameters and return them
    '''
    if label is None:
        label = "Crystal"
    if feature is None:
        feature = "area"
    max_out = 2.50*10**6 if feature == 'area' else 2000

    df = load_dataframe(row["rel_path"])
    histogram_min = df[feature].min()
    histogram_max = df[feature].max()
    histogram_bins = np.round(np.linspace(histogram_min, max_out, 100))
    df_oi = df[df["Labels"] == label]
    df_oi = df_oi.replace([np.inf,-np.inf], np.nan)
    df_oi = df_oi[df_oi[feature].notna()]
    counts = len(df_oi)
    if counts == 0:
        raise Exception(f"No {label} in {row.path}")
            # Schulz Plotting
    freq = get_frequency(df_oi,histogram_bins,feature)
    scale_factor = counts*histogram_bins[0]
    prob = np.array(freq)/(scale_factor)
    x_bins = histogram_bins + (histogram_bins[2]-histogram_bins[1])/2
    popt,pcov = curve_fit(schultz_zimm,x_bins,prob)
    a = popt[0]
    b = popt[1]
    molecular_weight = b/a
    molecular_number = (b+1)/a
    pdi = molecular_weight/molecular_number
    row['a'] = a
    row['b'] = b
    row['pdi'] = float(pdi)
    return row

def plot_feature_with_experiments_gaussian(
        df,
        feature:str='area',
        experiment_list:list[str]=[],
        color_list:list[str]=["b","g","m","r","y"],
        labels = ["Crystal","Multiple Crystal","Incomplete","Poorly Segmented"]
    ):
    '''
    Given a feature found in ML Result csvs, plot it for each experiment listed

    '''
    df_dict = {}
    for experiment in experiment_list:
        df_sub = df[df.search_str == experiment]
        df_dict[experiment] = df_sub

    # Clipping Variable
    max_out = 2.50*10**6 if feature == 'area' else 2000

    # Set up figure and axis layout
    fig,ax = plt.subplots(len(labels),1,figsize=(15,15) )
    fig.tight_layout(pad=5.0)
    fig.suptitle(f"Distributions of {feature}")
    histogram_min = min([df[feature].min() for _,df in df_dict.items()])
    histogram_max = max([df[feature].max() for _,df in df_dict.items()])
    histogram_bins = np.round(np.linspace(histogram_min, max_out, 100))

    for ii,(name,df) in enumerate(df_dict.items()):
        c_oi=color_list[ii]
        total_counts = len(df)
        for jj,label in enumerate(labels):
            ax_oi=ax[jj]
            df_oi = df[df["Labels"] == label]
            df_oi = df_oi.replace([np.inf,-np.inf], np.nan)
            df_oi = df_oi[df_oi[feature].notna()]
            counts = len(df_oi)
            if counts == 0:
                continue

            # Gaussian Plotting
            print(label)
            print(histogram_bins[0],len(df_oi))
            #freq = get_frequency(df_oi,histogram_bins,feature)
            freq,bin_edges = np.histogram(df_oi[feature],bins=100,range=(histogram_min,max_out))
            x_bins = bin_edges + (bin_edges[2]-bin_edges[1])/2
            p0 = [max(freq),max(x_bins)/10,(histogram_bins[2]-histogram_bins[1])*20]
            bounds = [(0,0,0,),
              (max(freq)*2,np.inf,np.inf)]
            popt,pcov = curve_fit(gaussian,x_bins[:-1],freq,
                                  p0=p0,
                                  bounds=bounds
                                  )
            mu = popt[1]
            sig = popt[2]

            ax_oi.plot(x_bins,gaussian(x_bins,*popt),color=c_oi,
                label=f"{name} ({counts}/{total_counts} mu: {mu:.3f}, sig: {sig:.3f}")
            ax_oi.hist(x=df_oi[feature],bins=histogram_bins,color=c_oi,alpha=.5)
            ax_oi.legend()
            ax_oi.set_title(label)
    return fig

def row_fit_gaussian(row:pd.Series,label=None,feature=None):
    '''
    Given a row from the formatted df(see next function), get the Gaussian Parameters
    '''
    if label is None:
        label = "Crystal"
    if feature is None:
        feature = "area"
    max_out = 2.50*10**6 if feature == 'area' else 2000

    df = load_dataframe(row["rel_path"])
    histogram_min = df[feature].min()
    histogram_max = df[feature].max()
    histogram_bins = np.round(np.linspace(histogram_min, max_out, 100))
    df_oi = df[df["Labels"] == label]
    df_oi = df_oi.replace([np.inf,-np.inf], np.nan)
    df_oi = df_oi[df_oi[feature].notna()]
    counts = len(df_oi)
    if counts == 0:
        raise Exception(f"No {label} in {row.path}")
            # Schulz Plotting
    freq = get_frequency(df_oi,histogram_bins,feature)
    scale_factor =1
    #prob = np.array(freq)/(scale_factor)
    x_bins = histogram_bins + (histogram_bins[2]-histogram_bins[1])/2
    p0 = [max(freq),max(x_bins)/10,(histogram_bins[2]-histogram_bins[1])*2]
    bounds = [(0,0,0,),
              (max(freq)*2,np.inf,np.inf)]
    popt,pcov = curve_fit(gaussian,x_bins,freq,p0=p0,
                          bounds=bounds,
                          maxfev = 100000000)
    A = popt[0]
    mu = popt[1]
    sig = popt[2]
    r2 = r2_score(freq,gaussian(x_bins,*popt))

    row["A"] = A
    row["mu"] = mu
    row["sig"] = sig
    row["r2"] = r2
    return row

def create_formatted_df(csv_list,
                        overwrite_string:str=None):
    '''
    Given a list of csv files, create dataframes w/ file information
    If overwrite string is used, use it for EVERY csv in the overwrite string list
    '''
    regex = "(?<=[_|\s])?([^_]+)-([^_]+)(?=[_|\s])?"
    
    df_arr = []
    for csv_path in csv_list:
        if overwrite_string is None:
            search_str = Path(csv_path).stem
        else:
            search_str = overwrite_string

        found = re.findall(regex,search_str)
        identifier_kwargs = {key:val for key,val in found}
        identifier_kwargs = identifier_kwargs | {"search_str":search_str,
                                                "path":csv_path
                                                }
        # Update these values into every column of the dataframe
        df_temp = pd.read_csv(csv_path)
        for identifier,val in identifier_kwargs.items():
            df_temp[identifier] = [val]*len(df_temp)

        df_arr.append(df_temp)
    
    df_final=pd.concat(df_arr)
    return df_final
