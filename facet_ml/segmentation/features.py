# Purpose of this file is to create features beyond
# those provided as part of skimage regionprops

import sys
from facet_ml.segmentation.segmenter import ImageSegmenter
import pandas as pd
import numpy as np
import copy



def facet_score(image_segmenter:ImageSegmenter):
    '''
    This function will add another column onto the image_segmenter's df
    This column will be the feature "facet_score" defined by
    (# Edge pixels in region)/(# Pixels in Region)*radial_equivalent
    Roughly, # edge pixels scales with r, # pixels scales r^2, hence the need for radial equivalent
    We would expect Crystals to have a high (but not too high) facet score,
    representing having some edges to facets but NOT too many!
    '''
    df = image_segmenter.df
    markers2 = image_segmenter.markers2

    def row_facet_score(row:pd.Series):
        marker_val = row.Region + image_segmenter._label_increment
        region_mask = (markers2 == marker_val)
        edges = copy.deepcopy(image_segmenter.img_edge)
        
        # Isolate the edge pixels
        edges[~region_mask] = 0
        edges[region_mask] = 1
        edge_score = np.sum(edges)

        # Get area of region
        area = np.sum(region_mask.astype(np.uint8))

        # Descale the major_axis_length and use that
        major_axis_length = row.major_axis_length/image_segmenter.pixels_to_um
        minor_axis_length = row.minor_axis_length/image_segmenter.pixels_to_um
        r_equivalent = np.sqrt((major_axis_length**2+minor_axis_length**2)/2) # Approximation
        return (edge_score/area)*r_equivalent
    print(len(df))
    df["facet_score"] = df.apply(row_facet_score,axis=1)

        
def merge_new_features(df_left,df_right,feature_to_merge,columns_to_merge_on):
    '''
    The goal of this is to merge feature_to_merge from df_right into df_left 
    This requires that both df have their columns to merge on identical 
    '''
    df_right_reduced = df_right[[feature_to_merge,*columns_to_merge_on]]
    df_adjusted = pd.merge(left=df_left,right=df_right_reduced,on=columns_to_merge_on)
    return df_adjusted
    raise NotImplemented
