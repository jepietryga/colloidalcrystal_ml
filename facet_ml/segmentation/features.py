# Purpose of this file is to create features beyond
# those provided as part of skimage regionprops

import sys
import pandas as pd
import numpy as np
import copy
from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty

from skimage.measure import find_contours
import itertools

class BaseRegionFeaturizer(ABC):

    @abstractmethod 
    def apply_featurizer(self):
        pass

    @abstractproperty
    def feature_name(self):
        pass

class Region():

    def __init__(self,
            region:np.ndarray,
            featurizers:list[BaseRegionFeaturizer]=[],
            
    ):
        '''
        A class that represents a distinct region within an image.

        This should provide easier visual reference of the region and thus be used to generate additional features

        Args:
            region (np.ndarray): The region masked out from the rest of the image
        '''
        # Default variables
        self.region = region
        self.featurizers = featurizers

        # Memoized variables
        self._edge_pixels = None
        self._curvature_values = None
        self._min_contour_length = None
        self._window_size_ratio = None

    @property
    def binary(self):
        return self.region > 0

    def featurize(self) -> dict:
        '''
        For each featurizer, run the self through it
        Gather the names and properties, return it as a dict
        '''
        feature_set = {}
        for featurizer in self.featurizers:
            property = featurizer.feature_name
            value = featurizer.apply_featurizer(self)
            feature_set[property] = value
        return feature_set

    @classmethod
    def from_image_and_mask(cls,
                            image:np.ndarray,
                            mask:np.ndarray):
        input_region = copy.deepcopy(image)
        input_region[~mask] = 0
        return cls(input_region)
    
    @classmethod
    def from_image_markers_num(cls,
                               image:np.ndarray,
                               markers:np.ndarray,
                               num:int):
        mask = (markers==num)
        input_region = copy.deepcopy(image)
        input_region[~mask] = 0
        return cls(input_region)

## Define Featurizers

class AverageCurvatureFeaturizer(BaseRegionFeaturizer):

    def __init__(self,
                 min_contour_length:int=20,
                 window_size_ratio:float=1/5
                 ):
        '''
        Get the average curvature values for a region
        '''
        self.min_contour_length = min_contour_length
        self.window_size_ratio = window_size_ratio

    def apply_featurizer(self, region:Region):
        '''
        Get the average curvature values for a region
        '''
        edge_pixels, curvature_values = get_region_curvatures(region,
                                    self.min_contour_length,
                                    self.window_size_ratio
                                )
        return np.mean(curvature_values)

    @property
    def feature_name(self):
        return "mean_curvature"
    
class MaxCurvatureFeaturizer(BaseRegionFeaturizer):

    def __init__(self,
                 min_contour_length:int=20,
                 window_size_ratio:float=1/5
                 ):
        '''
        Get the maximum curvature values for a region
        Motivation: A well-faceted crystal should have no extreme convexity values
        '''
        self.min_contour_length = min_contour_length
        self.window_size_ratio = window_size_ratio

    def apply_featurizer(self, region:Region):
        '''
        Get the maximum curvature values for a region
        '''
        edge_pixels, curvature_values = get_region_curvatures(region,
                                    self.min_contour_length,
                                    self.window_size_ratio
                                )
        if len(curvature_values) > 0:
            max = np.max(curvature_values)
        else:
            max= np.nan
        return max

    @property
    def feature_name(self):
        return "max_curvature"
    
class MinCurvatureFeaturizer(BaseRegionFeaturizer):

    def __init__(self,
                 min_contour_length:int=20,
                 window_size_ratio:float=1/5
                 ):
        '''
        Get the minimum curvature values for a region
        Motivation: A well-faceted crystal should have no extreme concavity values
        '''
        self.min_contour_length = min_contour_length
        self.window_size_ratio = window_size_ratio

    def apply_featurizer(self, region:Region):
        '''
        Get the minimum curvature values for a region
        '''
        edge_pixels, curvature_values = get_region_curvatures(region,
                                    self.min_contour_length,
                                    self.window_size_ratio
                                )
        if len(curvature_values) > 0:
            min = np.min(curvature_values)
        else:
            min = np.nan
        return min

    @property
    def feature_name(self):
        return "min_curvature"
    
class StdCurvatureFeaturizer(BaseRegionFeaturizer):

    def __init__(self,
                 min_contour_length:int=20,
                 window_size_ratio:float=1/5
                 ):
        '''
        Get the standard deviation of curvature values for a region
        Motivation: A well-faceted crystal should ben early bimodal and thus have some std
                    Too much is suspicious, though
        '''
        self.min_contour_length = min_contour_length
        self.window_size_ratio = window_size_ratio

    def apply_featurizer(self, region:Region):
        '''
        Get the standard deviation of curvature values for a region
        '''
        edge_pixels, curvature_values = get_region_curvatures(region,
                                    self.min_contour_length,
                                    self.window_size_ratio
                                )
        if len(curvature_values) > 0:
            std = np.std(curvature_values)
        else:
            std = np.nan
        return std

    @property
    def feature_name(self):
        return "std_curvature"
    
class PercentConvexityCurvatureFeaturizer(BaseRegionFeaturizer):

    def __init__(self,
                 min_contour_length:int=20,
                 window_size_ratio:float=1/5,
                 curvature_offset=0
                 ):
        '''
        Get the % of a region that is convex
        Motivation: Crystals should be nearly 100% convex, with lower convexity implying more divets
        '''
        self.min_contour_length = min_contour_length
        self.window_size_ratio = window_size_ratio
        self.curvature_offset = curvature_offset

    def apply_featurizer(self, region:Region):
        '''
        Get the % of a region that is convex
        '''
        edge_pixels, curvature_values = get_region_curvatures(region,
                                    self.min_contour_length,
                                    self.window_size_ratio
                                )
        if len(edge_pixels) == 0:
            return np.nan
        
        convex_paths, concave_paths = find_contiguous_curvatures(edge_pixels,curvature_values,self.curvature_offset)

        total_convexity_amt = np.sum([len(path) for path in convex_paths])
        total_convexity_percent = total_convexity_amt/len(edge_pixels) * 100
        return total_convexity_percent

    @property
    def feature_name(self):
        return "percent_convexity_curvature"
    
class LongestContiguousConvexityCurvatureFeaturizer(BaseRegionFeaturizer):

    def __init__(self,
                 min_contour_length:int=20,
                 window_size_ratio:float=1/5,
                 curvature_offset=0
                 ):
        '''
        Get the longest contiguous % of a region that is convex
        Motivation: Larger convex paths regular shapes, more crystalline
        
        '''
        self.min_contour_length = min_contour_length
        self.window_size_ratio = window_size_ratio
        self.curvature_offset = curvature_offset

    def apply_featurizer(self, region:Region):
        '''
        Get the longest contiguous % of a region that is convex
        '''
        edge_pixels, curvature_values = get_region_curvatures(region,
                                    self.min_contour_length,
                                    self.window_size_ratio
                                )
        if len(edge_pixels) == 0:
            return np.nan
        
        convex_paths, concave_paths = find_contiguous_curvatures(edge_pixels,curvature_values,self.curvature_offset)

        if len(convex_paths) > 0:
            longest_convex_path = max(convex_paths, key=lambda x:len(x))
        else:
            longest_convex_path = []
        contiguous_convex_path_percent = len(longest_convex_path)/len(edge_pixels) * 100
        return contiguous_convex_path_percent

    @property
    def feature_name(self):
        return "longest_contiguous_percent_convexity_curvature"
    
class LongestContiguousConcavityCurvatureFeaturizer(BaseRegionFeaturizer):

    def __init__(self,
                 min_contour_length:int=20,
                 window_size_ratio:float=1/5,
                 curvature_offset=0
                 ):
        '''
        Get the longest contiguous % of a region that is convex
        Motivation: Larger concave paths imply larger irregular divets, less crystal-like
        '''
        self.min_contour_length = min_contour_length
        self.window_size_ratio = window_size_ratio
        self.curvature_offset = curvature_offset

    def apply_featurizer(self, region:Region):
        '''
        Get the longest contiguous % of a region that is convex
        '''
        edge_pixels, curvature_values = get_region_curvatures(region,
                                    self.min_contour_length,
                                    self.window_size_ratio
                                )
        if len(edge_pixels) == 0:
            return np.nan
        
        convex_paths, concave_paths = find_contiguous_curvatures(edge_pixels,curvature_values,self.curvature_offset)

        if len(concave_paths) > 0:
            longest_concave_path = max(concave_paths, key=lambda x:len(x))
        else:
            longest_concave_path = []
        contiguous_concave_path_percent = len(longest_concave_path)/len(edge_pixels) * 100
        return contiguous_concave_path_percent

    @property
    def feature_name(self):
        return "longest_contiguous_percent_concavity_curvature"
    
class DistinctPathsCurvatureFeaturizer(BaseRegionFeaturizer):

    def __init__(self,
                 min_contour_length:int=20,
                 window_size_ratio:float=1/5,
                 curvature_offset=0
                 ):
        '''
        Get the # paths needed to describe concavity and convexity
        Motivation: More paths needed implies stranger shapes, less likely a crystal
        '''
        self.min_contour_length = min_contour_length
        self.window_size_ratio = window_size_ratio
        self.curvature_offset = curvature_offset

    def apply_featurizer(self, region:Region):
        '''
        Get the % of a region that is convex
        '''
        edge_pixels, curvature_values = get_region_curvatures(region,
                                    self.min_contour_length,
                                    self.window_size_ratio
                                )
        if len(edge_pixels) == 0:
            return np.nan

        convex_paths, concave_paths = find_contiguous_curvatures(edge_pixels,curvature_values,self.curvature_offset)

        return len(convex_paths) + len(concave_paths)

    @property
    def feature_name(self):
        return "number_distinct_paths_curvature"

def get_region_curvatures(region:Region, 
                          min_contour_length:int=5,
                          window_size_ratio:float=1/5):
    '''
    Given a region, get the curvatures of each edge pixel.

    Original code from: https://medium.com/@stefan.herdy/compute-the-curvature-of-a-binary-mask-in-python-5087a88c6288

    Args:
        min_contour_length (int) : From the ffound contours, it must be at least this size
        window_size_ratio (float) : Ratio of how much % curve is a pixel's neighborhood for curvature calc
    '''
    # If values are stored...
    if not isinstance(region._edge_pixels,type(None)):
        # and functions variables match up...
        if region._min_contour_length == min_contour_length and region._window_size_ratio == window_size_ratio:
            # Memoized
            return region._edge_pixels, region._curvature_values

    contours = find_contours(region.region)
    # Initialize arrays to store the curvature information for each edge pixel
    curvature_values = []
    edge_pixels = []
    
    for contour in contours:
        # Iterate over each point in the contour
        for i, point in enumerate(contour):
            # We set the minimum contour length to 20
            # You can change this minimum-value according to your specific requirements
            if contour.shape[0] > min_contour_length:
                # Compute the curvature for the point
                # We set the window size to 1/5 of the whole contour edge. Adjust this value according to your specific task
                window_size = int(contour.shape[0]*window_size_ratio)
                
                #window_size = 50
                curvature = compute_curvature(point, i, contour, window_size)
                # We compute, whether a point is convex or concave.
                # If you want to have the 2nd derivative shown you can comment this part
                #if curvature > 0:
                #    curvature = 1
                #if curvature <= 0:
                #    curvature = -1
                # Store curvature information and corresponding edge pixel
                curvature_values.append(curvature)
                edge_pixels.append(point)

    # Convert lists to numpy arrays for further processing
    curvature_values = np.array(curvature_values)
    edge_pixels = np.array(edge_pixels)

    # Memoize for later in the region
    region._edge_pixels = edge_pixels
    region._curvature_values = curvature_values
    region._min_contour_length = min_contour_length
    region._window_size_ratio = window_size_ratio

    return edge_pixels, curvature_values

def compute_curvature(point, i, contour, window_size):
    '''
    Compute the curvature using polynomial fitting in a local coordinate system
    
    Args:
        point (tuple): (x,y) tuple of a point
        i (int): index of the point in the total contour
        contour (np.ndarray): All points that make up a contour
        window_size (int): Size of the neighborhood of the contour
    '''
    # Extract neighboring edge points
    start = max(0, i - window_size // 2)
    end = min(len(contour), i + window_size // 2 + 1)
    neighborhood = contour[start:end]

    # Extract x and y coordinates from the neighborhood
    x_neighborhood = neighborhood[:, 1]
    y_neighborhood = neighborhood[:, 0]

    # Compute the tangent direction over the entire neighborhood and rotate the points
    tangent_direction_original = np.arctan2(np.gradient(y_neighborhood), np.gradient(x_neighborhood))
    tangent_direction_original.fill(tangent_direction_original[len(tangent_direction_original)//2])

    # Translate the neighborhood points to the central point
    translated_x = x_neighborhood - point[1]
    translated_y = y_neighborhood - point[0]


    # Apply rotation to the translated neighborhood points
    # We have to rotate the points to be able to compute the curvature independent of the local orientation of the curve
    rotated_x = translated_x * np.cos(-tangent_direction_original) - translated_y * np.sin(-tangent_direction_original)
    rotated_y = translated_x * np.sin(-tangent_direction_original) + translated_y * np.cos(-tangent_direction_original)

    # Fit a polynomial of degree 2 to the rotated coordinates
    coeffs = np.polyfit(rotated_x, rotated_y, 2)


    # You can compute the curvature using the formula: curvature = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
    # dy_dx = np.polyval(np.polyder(coeffs), rotated_x)
    # d2y_dx2 = np.polyval(np.polyder(coeffs, 2), rotated_x)
    # curvature = np.abs(d2y_dx2) / np.power(1 + np.power(dy_dx, 2), 1.5)

    # We compute the 2nd derivative in order to determine whether the curve at the certain point is convex or concave
    curvature = np.polyval(np.polyder(coeffs, 2), rotated_x)

    # Return the mean curvature for the central point
    return np.mean(curvature)

def find_contiguous_curvatures(edge_pixels:np.ndarray,
                               curvature_values:np.ndarray,
                               curvature_offset:float=0.0):
    '''
    Find the stretches of pixels that are above or below 0 (convex or concave)
    Use offset to include or disclude straightaways in the curvature
    '''

    convex_logical = curvature_values >= curvature_offset

    convex_paths = []
    concave_paths = []

    for convex, val in itertools.groupby(enumerate(edge_pixels),key=lambda x:convex_logical[x[0]]):
        _, edge_pixels = zip(*val)
        if convex:
            convex_paths.append(edge_pixels)
        else:
            concave_paths.append(edge_pixels)

    # For each of the paths, check the edge case
    # that the last path is continuous with the first
    if len(convex_paths) > 1:
        first_path = convex_paths[0]
        last_path = convex_paths[-1]
        # Same path
        if np.equal(last_path[-1],first_path[0]).all():
            new_last = tuple([*last_path,*first_path])
            convex_paths = convex_paths[1:-1] + [new_last]

    if len(concave_paths) > 1:
        first_path = concave_paths[0]
        last_path = concave_paths[-1]
        # Same path
        if np.equal(last_path[-1],first_path[0]).all():
            new_last = tuple([*last_path,*first_path])
            concave_paths = concave_paths[1:-1] + [new_last]

    return convex_paths, concave_paths

def facet_score(image_segmenter):
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
