# This is an improvement to the segmentation_utils class to make it more general

import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import math
import copy
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
from cached_property import cached_property
from IPython.display import clear_output
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from functools import partial
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
import pickle

class ImageSegmenter():
    
    def __init__(self,
                input_path=None,
                pixels_to_um=9.37,
                top_boundary=0,
                bottom_boundary=960,
                left_boundary=0,
                right_boundary=2560,
                result_folder_path="../Results",
                override_exists=False,
                threshold_mode = "otsu",
                edge_modification = None,
                file_str = None
                ):
        '''
        Args:
            input_path (string OR img)    : Path to the image desired to be interpreted (if img, create tmp file)
            pixels_to_um (float)   : Scale factor for image analysis
            top_boundary (int)     : Pixel boundary for cropping image
            bottom_boundary (int)  : Pixel boundary for cropping image
            left_boundary (int)    : Pixel boundary for cropping image
            right_boundary (int)   : Pixel boundary for cropping image
            result_folder_path (string) : Path to the folder .csv should be saved.
            override_exists (bool) : If .csv already exists, DO NOT overwrite it if this variable is False. Allows classification across sessions
        '''
        self.input_path = input_path
        self.pixels_to_um = pixels_to_um
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.override_exists = override_exists
        self.threshold_mode = threshold_mode
        self.edge_modification = edge_modification

        # Derived Variables
        if isinstance(input_path,str):
            self._file_name = '.'.join(self.input_path.split('/')[-1].split('.')[:-1])
        else:
            self.input_path = 'tmp.png'
            self._file_name = 'tmp'
            cv2.imwrite(f'{self._file_name}.png', input_path, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
        os.makedirs(result_folder_path,exist_ok=True)
        self._csv_file  = f'{result_folder_path}/values_{self._file_name}_{file_str}.csv'
            
        # Define default image variables
        # NOTE: Will need to do something with this in the future for input
        self.canny_tl = 40
        self.canny_tu = 40
        self.blur_size = (5,5)
        self.kernel = np.ones((3,3),np.uint8) 
        self.distance_scale = .35 # For controlling distance transform
        self.bilateral_d = 50
        self.bilateral_ss = 90

        # Custom Pixel Classifier Variables
        self.pixel_model = None

        self.process_images(edge_modification=self.edge_modification)

        #print(f'Image Segmenter on {self._file_name} created!')

    def process_images(self,
        blur=False,
        edge_modification=False,
        use_bilateral=False):
        '''
        Main runner for creating images
        '''
        self.img  = cv2.imread(self.input_path, 0)
        self.img2 = self.img[self.top_boundary:self.bottom_boundary,
                                self.left_boundary:self.right_boundary]
        self.img3 = cv2.imread(self.input_path, 1)
        self.img3 = self.img3[self.top_boundary:self.bottom_boundary,
                                self.left_boundary:self.right_boundary]

        self.set_markers(blur=blur,edge_modification=edge_modification,use_bilateral=use_bilateral)

        self.regions_list = np.unique(self.markers)-self.label_increment
        #print(self.regions_list)
        self.regions_list = [x for x in self.regions_list if x > 0]

        #self.img3[self.markers2 == -1] = [0, 255,255]

        self.img4 = color.label2rgb(self.markers2, bg_label=0)
        self.decorate_regions()

    def set_markers(self,
        blur=False,
        edge_modification=False,
        use_bilateral=False):
        '''Perform Watershed algorithm, return markers'''
        #Setting up markers for watershed algorithm

        kernel = self.kernel

        self._generate_threshold()
        self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, kernel,iterations = 3)

        #what is definitely your background?
        self._bg_mark = cv2.dilate(self.thresh,kernel)

        #apply distance transform
        if edge_modification:
            self._perform_edge_modification() # Adds "background" for dist transform to catch 
        
        # Add 0 border, distance transform, remove 0 border
        thresh_border = cv2.copyMakeBorder(self.thresh,
                                            top=1,
                                            bottom=1,
                                            right=1,
                                            left=1,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=0
                                            )
        self._dist_transform = cv2.distanceTransform(thresh_border, cv2.DIST_L2, 5)
        self._dist_transform = self._dist_transform[1:-1,1:-1]

        #thresholding the distance transformed image
        ret2, fg_mark = cv2.threshold(self._dist_transform, self.distance_scale*self._dist_transform.max(), 255, 0)
        self._fg_mark = np.uint8(fg_mark)

        #the unknown pixels
        self.unknown = cv2.subtract(self._bg_mark, self._fg_mark)

        self.outputs = cv2.connectedComponentsWithStats(self._fg_mark)
        self.label_increment = 10

        self.markers = self.outputs[1]+self.label_increment

        self.markers[self.unknown == 255]=0
        temp_markers = copy.deepcopy(self.markers)
        self.markers2 = cv2.watershed(self.img3,temp_markers)

    def _otsu_threshold(self,blur=None):
        '''
        Otsu thresholding
        '''
        threshable = self.img2 if not blur else cv2.GaussianBlur(self.img2, self.blur_size,0)
        ret, thresh = cv2.threshold(threshable, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return ret, thresh

    def _pixel_threshold(self):
        '''
        Use Weka-inspired pixel segmenter to generate threshold
        NOTE: Still developing best model for this.
        '''
        # Load model
        self.load_pixel_segmenter()

        # Featurize Image
        sigma_min = 1
        sigma_max = 16
        features_func = partial(feature.multiscale_basic_features,
                        intensity=False, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        channel_axis=None)
        features = features_func(self.img2)

        # Flatten features
        (x,y,z) = features.shape
        features = features.reshape(x*y,z)

        # Predict
        results = future.predict_segmenter(features,self.pixel_model)

        # Reshape
        thresh = 255*(results.reshape(x,y).astype(np.uint8)-1) # To make background and not
        return thresh

    def _local_threshold(self):
        '''
        Use Adaptive (local) threhsolding
        '''
        thresh = cv2.adaptiveThreshold(self.img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,53,10)
        return thresh

    def _generate_threshold(self,blur=None,threshold_mode=None):
        '''
        Method of creating threshold, uses different modes
        '''
        if threshold_mode == None:
            threshold_mode = self.threshold_mode
        if threshold_mode == "otsu":
            self.ret, self.thresh = self._otsu_threshold(blur)
            return

        if threshold_mode == "pixel":
            self.thresh = self._pixel_threshold()
            return 
        if threshold_mode == "local":
            self.thresh = self._local_threshold()
            return
        
        if threshold_mode == "ensemble":
            thresh_otsu = self._otsu_threshold(blur)[1].astype(bool)
            thresh_local = self._local_threshold().astype(bool)
            thresh_pixel = self._pixel_threshold().astype(bool)
            ensemble = ( (thresh_pixel) & (thresh_local | thresh_otsu)) \
                        | (thresh_otsu & thresh_local)
            self.thresh = ensemble.astype(np.uint8)*255
            return
        
    def _perform_edge_modification(self, edge_modification = None):
        '''
        Perform edge modification of threshold using internal schemes

        dark_bright: Use dark_bright edge detection scheme
        '''
        if not edge_modification:
            edge_modification = self.edge_modification
            #print(edge_modification)

        if edge_modification == None:
            return
        elif edge_modification == "dark_bright": # Note: Probably add this as separate utility
            # Get Canny Edge
            canny_args = [(3,3),80,160,80,80,False]
            canny_edge = self.canny_edge(*canny_args)

            # Define sharpen kernel
            n = -1
            m = 9
            kernel = np.array([[n,n,n],
                            [n,m,n],
                            [n,n,n]]
                            )*1
            kernel = np.ones([7,7])*n
            kernel[3,3] = 49

            # Get just edge intensities
            img_edges = cv2.filter2D(self.img2,-1,kernel)
            img_edges[canny_edge == 0] = 0

            # Make histograms for brightness/darkness heuristic
            histogram, bin_edges = np.histogram(img_edges,bins=256)

            # remove 0 vals, 255 vals
            bin_edges = bin_edges[1:-1]
            histogram = histogram[1:-1]
            cut_off = bin_edges[np.argmax(histogram)]

            # Define bright edges
            bright_edges = copy.deepcopy(img_edges)
            bright_edges[bright_edges < cut_off] = 0
            bright_edges[bright_edges > 0] = 255

            # define dark edges
            dark_edges = copy.deepcopy(img_edges)
            dark_edges[dark_edges > cut_off] = 0
            dark_edges[dark_edges > 0] = 255

            # broaden edges for visibility, store for figure reference
            bright_broad = cv2.GaussianBlur(bright_edges,(3,3),cv2.BORDER_DEFAULT)
            dark_broad  = cv2.GaussianBlur(dark_edges,(3,3),cv2.BORDER_DEFAULT)
            color_img = cv2.cvtColor(self.img2,cv2.COLOR_GRAY2RGB)
            color_img[bright_broad > 0] = (0,0,255)
            color_img[dark_broad > 0] = (0,255,0)
            #color_img[(bright_broad > 0) & (dark_broad>0)] = (255,0,0)
            self._edge_highlight = color_img
            self._dark_edges = dark_edges
            self._original_thresh = copy.deepcopy(self.thresh)

            # Modify threshold
            self.thresh = self.thresh-dark_edges

    def load_pixel_segmenter(self):
        '''
        Load the pixel classifer. Is a LARGE model, so only use this if needed
        '''
        if not self.pixel_model:
                with open("../Models/bg_segmenter.pickle","rb") as f:
                    self.pixel_model = pickle.load(f)

    def decorate_regions(self):
        '''
        Labels image 4 using information from Scikit to ensure commensurate labeling
        NOTE: The big issue is ensuring regions line up
        '''
        
        for i in self.regions_list:
            cx= int(self.outputs[3][i][0])
            cy= int(self.outputs[3][i][1])
            cv2.putText(self.img4, text= str(i), org=(cx,cy),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0),
                    thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(self.img4, (cx, cy), radius = 3, color=(0,0,0), thickness=-1)
        return


    @cached_property # NOTE: Cached Property is basically a memoized function
    def df(self): # Might not actually want this to be cached if we're making this interactive...
        file_present = os.path.isfile(self._csv_file)
        if file_present and not self.override_exists:
            df = pd.read_csv(self._csv_file)
            self.number_labels = len(df['area'])
            return pd.read_csv(self._csv_file)

        propList = ['area',
            'equivalent_diameter', 
            'orientation', 
            'major_axis_length',
            'minor_axis_length',
            'perimeter',
            'min_intensity',
            'mean_intensity',
            'max_intensity',
            'solidity',
            'eccentricity',
            'centroid_local',
            'feret_diameter_max',
            'moments',
            'moments_central',
            'moments_hu',
            'label'
            ]
        clusters = measure.regionprops_table(self.markers2-self.label_increment, self.img2,properties=propList)

        scaled_features = ['equivalent_diameter',
                           'major_axis_length',
                           'minor_axis_length',
                           'perimeter',
                           'feret_diameter_max',
                           #'solidity'
                          ]
        for key,val in clusters.items():
            #print(f'{key}: {len(val)}')
            if key == 'area':
                clusters[key] = clusters[key]*self.pixels_to_um**2
            if key == 'orientation':
                continue # Line didn't seem used to me previously...?
            if key == 'label':
                continue
            elif key in scaled_features:
                clusters[key] = clusters[key]*self.pixels_to_um

        # Add in Composite variables
        clusters['major_axis_length/minor_axis_length'] = clusters['major_axis_length']/clusters['minor_axis_length']
        clusters['perimeter/major_axis_length'] = clusters['perimeter']/clusters['major_axis_length']
        clusters['perimeter/minor_axis_length'] = clusters['perimeter']/clusters['minor_axis_length']

        # Add in Label, Filename, Region Columns
        self.number_labels = len(clusters['area'])
        labeling_list = [None] * self.number_labels
        filename_list = [self.input_path] * self.number_labels
        #print(clusters['label'])
        clusters['Labels'] = labeling_list
        clusters['Filename'] = filename_list
        clusters['Region'] = clusters['label']

        # Create CSV (override_exists is a safety variable to avoid rewriting data)
        return pd.DataFrame(clusters)

    def create_csv(self):
        if self.override_exists:
            self.df.to_csv(self._csv_file)
        else:
            print("WARNING: Override not in place")

    def _grab_region(self,img,label_oi,alpha=.75,buffer=20):
        mod_image = copy.deepcopy(img)
        label_marker = copy.deepcopy(self.markers2)
        label_marker[self.markers2 != label_oi] = 0

        y1 = grab_bound(label_marker,"top",buffer)
        y2 = grab_bound(label_marker,"bottom",buffer)
        x1 = grab_bound(label_marker,"left",buffer)
        x2 = grab_bound(label_marker,"right",buffer)

        mod_image[label_marker != label_oi] = mod_image[label_marker != label_oi]*alpha
        return mod_image[y1:y2,x1:x2]

    @cached_property
    def region_arr(self):
        '''
        img Regions associated with the image segmentation 
        NOTE: _Slightly_ different from 'begin_labeling' as we remove non-region ENTIRELY
        '''
        return self.grab_region_array(focused=True)

        self.df # Make sure this is initiated
        data_arr = []
        ii = 0
        regions_list = list(self.df["Region"])
        
        while ii < len(regions_list): # 1-Offset for counting purposes
            region_oi = regions_list[ii]+self.label_increment

            data_arr.append(self._grab_region(self.img2,region_oi,alpha=0,buffer=5))
            ii += 1
        return data_arr
    
    def grab_region_array(self,focused=True):
        '''
        Grab an array of images that are bounded (focused) or the same size as img2 (nopt focused)
        Can be useful for quickly making bools of regions
        '''
        self.df # Make sure this is initiated
        data_arr = []
        ii = 0
        regions_list = list(self.df["Region"])
        
        while ii < len(regions_list): # 1-Offset for counting purposes
            region_oi = regions_list[ii]+self.label_increment
            if focused:
                data_arr.append(self._grab_region(self.img2,region_oi,alpha=0,buffer=5))
            if not focused:
                data_arr.append(self._grab_region(self.img2,region_oi,alpha=0,buffer=np.inf))
            ii += 1
        return data_arr

    def begin_labeling(self):
        self.df # To ensure it's been initialized
        ii = 0
        regions_list = self.df["Region"]
        while ii < len(regions_list): # 1-Offset for counting purposes
            clear_output(wait=False)
            region_oi = regions_list[ii] # +3 gets past Borders and BG labeling

            testImage = self._grab_region(self.img3,region_oi+self.label_increment,alpha=.75,buffer=20)
            plt.figure(figsize = (10,10))
            plt.imshow(testImage)
            plt.show()
            
            # User Input
            input_list = ['C','M','P','I','B','D']
            
            print(f'Region {region_oi} (Max: {max(regions_list)}) \nNOTE: Skipping a region may mean a bad region was encountered\n')
            print("Type an integer to jump to region, or a character below to label image\n", 
                "C = Crystal, M = Multiple Crystal, P = Poorly Segmented, I = Incomplete, B = Back, D = Done")
            user_input = input()
            while (user_input not in input_list) and (not user_input.isnumeric()):
                user_input = input("Invalid Input, Retry: ")
            if user_input == 'B':
                ii = ii - 1
                continue
            elif user_input.isnumeric():
                ii = int(user_input) - 1 # Because 1-Offset
                continue;
            elif user_input == 'D':
                break;
                
            # Clean-up
            translated_input = None
            if user_input == 'C':
                translated_input = 'Crystal'
            elif user_input == 'M':
                translated_input = 'Multiple Crystal'
            elif user_input == 'P':
                translated_input = 'Poorly Segmented'
            elif user_input == 'I':
                translated_input = 'Incomplete'
            self.df.loc[self.df['Region'] == region_oi,'Labels'] = translated_input
            self.df.to_csv(self._csv_file)
            
            ii = ii + 1

    def labeling_mapping(self):
        '''
            Code added 2022.08.12 for debugging and salvaging data
        '''
        self.df # To ensure it's been initialized
        ii = 0
        regions_list = self.df["Region"]
        mapping_index = []
        mapping_region = []

        while ii < len(regions_list): # 1-Offset for counting purposes
            clear_output(wait=False)
            region_oi = regions_list[ii] # +3 gets past Borders and BG labeling
            
            #self.df[self.df['Region'] == region_oi]['Labels'] = translated_input
            mapping_index.append(ii)
            mapping_region.append(region_oi)
            ii = ii + 1
        return mapping_region,mapping_index

    def canny_edge(self, blur_size = None, tl = None,tu = None,d = None, ss = None, use_bilateral=False,):
        if not blur_size:
            blur_size = self.blur_size
        if not tl:
            tl = self.canny_tl
        if not tu:
            tu = self.canny_tu
        if not d:
            d = self.bilateral_d
        if not ss:
            ss = self.bilateral_ss

        if not use_bilateral:
            print(blur_size)
            img_blur = cv2.GaussianBlur(self.img2, blur_size,0)
        else:
            img_blur = cv2.bilateralFilter(self.img2,d,ss,ss)
        self.edge = cv2.Canny(img_blur,tl,tu)
        return self.edge

    


def grab_bound(img,mode="top",buffer=0):
    '''
    For an intensity img with region of interest and all others blacked out, get a bound defined by mode
    
    Returns x- or y-coordinate
    '''
    def bounded_expansion(coord,img,axis):
        if coord < 0:
            return 0
        elif coord > np.shape(img)[axis]:
            return np.shape(img)[axis]
        else:
            return coord

    if mode == "top":
        for yy in np.arange(0,np.shape(img)[0]):
            num_list = np.unique(img[yy,:])
            if len(num_list) > 1:
                return bounded_expansion(yy-buffer,img,0)
        
    elif mode == "bottom":
        for yy in np.arange(0,np.shape(img)[0])[::-1]:
            num_list = np.unique(img[yy,:])
            if len(num_list) > 1:
                return bounded_expansion(yy+buffer,img,0)
            
    elif mode == "left":
        for xx in np.arange(0,np.shape(img)[1]):
            num_list = np.unique(img[:,xx])
            if len(num_list) > 1:
                return bounded_expansion(xx-buffer,img,1)
        
    elif mode == "right":
        for xx in np.arange(0,np.shape(img)[1])[::-1]:
            num_list = np.unique(img[:,xx])
            if len(num_list) > 1:
                return bounded_expansion(xx+buffer,img,1)
    return -1