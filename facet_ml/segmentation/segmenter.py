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
from skimage.filters import threshold_local
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path
from typing import Union
import h5py

from facet_ml.segmentation import edge_modification as em
from facet_ml.segmentation import thresholding
from facet_ml.static.path import STATIC_MODELS
from facet_ml.segmentation import features as feat


class ImageSegmenter():
    
    def __init__(self,
                input_path=None,
                pixels_to_um=9.37,
                top_boundary=0,
                bottom_boundary=860,
                left_boundary=0,
                right_boundary=2560,
                result_folder_path="../../Results",
                override_exists=False,
                threshold_mode:Union[callable,str] = "otsu",
                edge_modification:Union[callable,str] = None,
                file_str = None,
                sam_kwargs = {
                    "points_per_side":64
                },
                region_featurizers = [
                        feat.AverageCurvatureFeaturizer(),
                        feat.StdCurvatureFeaturizer(),
                        feat.MinCurvatureFeaturizer(),
                        feat.MaxCurvatureFeaturizer(),
                        feat.PercentConvexityCurvatureFeaturizer(),
                        feat.LongestContiguousConcavityCurvatureFeaturizer(),
                        feat.LongestContiguousConvexityCurvatureFeaturizer(),
                        feat.DistinctPathsCurvatureFeaturizer()
                ]
                ):
        '''
        Main class for handling segmentation pipeline.

        Args:
            input_path (string OR img)    : Path to the image desired to be interpreted (if img, create tmp file)
            pixels_to_um (float)   : Scale factor for image analysis and featurization
            top_boundary (int)     : Pixel boundary for cropping image
            bottom_boundary (int)  : Pixel boundary for cropping image
            left_boundary (int)    : Pixel boundary for cropping image
            right_boundary (int)   : Pixel boundary for cropping image
            result_folder_path (string) : Path to the folder .csv should be saved.
            override_exists (bool) : If .csv already exists, DO NOT overwrite it if this variable is False. Allows classification across sessions
        '''
        # Given variables (besides input path, handle that at very end)
        self._input_path = input_path
        self.pixels_to_um = pixels_to_um
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.result_folder_path = result_folder_path
        self.override_exists = override_exists
        self.threshold_mode = threshold_mode
        self.edge_modification = edge_modification
        self.file_str = file_str
        self.sam_kwargs = sam_kwargs
        self.region_featurizers = region_featurizers

        # Image variables
        self._image_read = None
        self._image_cropped = None
        self._image_working = None
        self._image_labeled = None
        self._thresh = None

        os.makedirs(self.result_folder_path,exist_ok=True)
            
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
        self.edge_model = None

        # hidden variables
        self._img_edge = None
        self._edge_highlight = None
        self._live_edges = None
        self._dist_transform = None
        self._label_increment = 20
        self._df = None
        self._region_arr = None
        self._region_dict = None

        # PyQt variables
        self._region_tracker = None # Used for keeping tabs on where in region list we are, created in self.df

        # Segment Anything Model variables
        self._mask_generator = None

        # Input path stuff
        # Input Path
        self.input_path = self._input_path

    @property
    def input_path(self):
        return self._input_path

    @input_path.setter
    def input_path(self,value):
        self._input_path = value
        if self._input_path is not None:
            # Clear the dataframe and labeled_image, if it exists
            self._df = None
            self._image_read = None
            self._image_cropped = None
            self._image_working = None
            self._image_labeled = None
            self._thresh = None

            # Redefine internal paths (these may be removed at some point)
            if isinstance(self._input_path,str):
                self._file_name = '.'.join(self.input_path.split('/')[-1].split('.')[:-1])
            else:
                self.input_path = 'tmp.png'
                self._file_name = 'tmp'
                cv2.imwrite(f'{self._file_name}.png', self._input_path, [cv2.IMWRITE_PNG_COMPRESSION, 0])
           
            self._csv_file  = f'{self.result_folder_path}/values_{self._file_name}_{self.file_str}.csv'

            # self.process_images(edge_modification=self.edge_modification)

    @property
    def image_read(self):
        if self._image_read is None:
            self.process_images()
        return self._image_read
    
    @property
    def image_cropped(self):
        if self._image_cropped is None:
            self.process_images()
        return self._image_cropped
    
    @property
    def image_working(self):
        if self._image_working is None:
            self.process_images()
        return self._image_working
    
    @property
    def image_labeled(self):
        if self._image_labeled is None:
            self.process_images()
        return self._image_labeled
    
    @property
    def thresh(self):
        if self._thresh is None:
            self.process_images()
        return self._thresh

    def process_images(self,
        blur=False,
        edge_modification=False,
        use_bilateral=False):
        '''
        Create each of the images of interest.
        Performs segmentation as part of the process
        '''
        if self.input_path is None:
            raise Exception("Error: ImageSegmenter has no input_path")
        # Raw Read-in
        self._image_read  = cv2.imread(self.input_path, 0)
        self._image_cropped = self.image_read[self.top_boundary:self.bottom_boundary,
                                self.left_boundary:self.right_boundary]
        self._image_working = cv2.imread(self.input_path, 1)
        self._image_working = self.image_working[self.top_boundary:self.bottom_boundary,
                                self.left_boundary:self.right_boundary]

        self._thresh = np.full_like(self.image_cropped,0)

        # Perform segmentation
        self.set_markers(edge_modification=edge_modification)

        # Set regions by number, non-inclusive of background and edge border
        self.regions_list = np.unique(self.markers)-self._label_increment
        self.regions_list = [x for x in self.regions_list if x > 0]

        self._image_labeled = color.label2rgb(self.markers2, bg_label=0)
        #self.decorate_regions()

    def set_markers(self,
        edge_modification=False,):
        '''
        Create and set markers via segmentation method of choice
        Process will threshold, perform morphologoical operations, and then watershed
        '''
        

        # If using detectron, the threshold creation scheme is NOT NEEDED
        if self.threshold_mode == "detectron2":
            # Note to self: skimage.measure.label to leverage detectron2 model as a labeler
            masks = detectron2_maskrcnn_solids(self.image_working)
            self.markers2 = self._label_increment*np.ones(np.shape(self.image_cropped))
            num_markers,_,_ = np.shape(masks)
            for ii in np.arange(num_markers):

                mask_oi = masks[ii,:,:].astype(bool)
                mask_bulk = cv2.erode(mask_oi.astype(np.uint8),kernel=np.ones((3,3))).astype(bool)
                mask_edge = (~mask_bulk & mask_oi)
                self.markers2[mask_edge] = -1
                self.markers2[mask_bulk] = 1+self._label_increment+ii
            self.markers = copy.deepcopy(self.markers2).astype(int)
            self.markers2 = self.markers2.astype(int)
            return
        
        if self.threshold_mode == "segment_anything":
            if self._mask_generator == None:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
                model_type = "vit_l"
                sam_checkpoint = STATIC_MODELS["segment_anything_vit_l"]
                device = "cpu"
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=device)
                mask_generator = SamAutomaticMaskGenerator(sam,
                                                           **self.sam_kwargs)
                self._mask_generator = mask_generator
            masks = self._mask_generator.generate(self.image_working)
            
            self.markers2 = self._label_increment*np.ones(np.shape(self.image_cropped)) 
            for ii,mask in enumerate(masks):
                mask_oi = mask["segmentation"]
                mask_bulk = cv2.erode(mask_oi.astype(np.uint8),kernel=np.ones((3,3))).astype(bool)
                mask_edge = (~mask_bulk & mask_oi)
                self.markers2[mask_edge] = -1
                self.markers2[mask_bulk] = 1+self._label_increment+ii
            self.markers = copy.deepcopy(self.markers2).astype(int)
            self.markers2 = self.markers2.astype(int)
            return

        ## Threshold and Get background
        kernel = self.kernel

        self._thresh = self._generate_threshold()
        self._thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel,iterations = 2)
        self._thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, kernel,iterations = 1)
        
        # Sure Background
        self._bg_mark = cv2.dilate(self.thresh,kernel,iterations=1)
        self._pre_thresh = copy.deepcopy(self.thresh)
        
        if edge_modification:
            self._perform_edge_modification() # Adds "background" for dist transform to catch 
        
        ## Distance transform and foreground
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

        # Sure foreground
        scaling_rule = self._dist_transform.max()*.35
        ret2, fg_mark = cv2.threshold(self._dist_transform, scaling_rule, 255, 0)
        fg_mark = cv2.erode(fg_mark,kernel)
        self._fg_mark = np.uint8(fg_mark)


        ## Develop unknown region
        self.unknown = cv2.subtract(self._bg_mark, self._fg_mark)

        # Develop Regions
        self.outputs = cv2.connectedComponentsWithStats(self._fg_mark)
        self.markers = self.outputs[1]+self._label_increment
        self.markers[self.unknown == 255]=0
        temp_markers = copy.deepcopy(self.markers)
        #print(np.shape(self.image_cropped),np.shape(self.image_working))
        image_working_blur = cv2.GaussianBlur(self.image_working,(9,9),0) # NOTE: What's the impact of this
        self.markers2 = cv2.watershed(image_working_blur,temp_markers)
        
    def _clean_markers2(self):
        '''
        TESTING
        Method should be able to smooth out issues w/ Markers by individually modifying them
        May not work well with large data
        '''

        # NOTE: need to recreated background for each marker, too!
        marker2_working = self.markers2.copy()
        for region_oi in self.df.Region.unique():
            marker_oi = region_oi + self._label_increment
            marker_mask = self.markers2.copy()
            marker_logical = marker_oi == marker_mask
            marker_mask[~marker_logical] = 0
            marker_mask[marker_logical] = 255

            kernel = np.ones([3,3])
            marker_mask = cv2.erode(marker_mask.astype(np.uint8),kernel,iterations=5)
            marker_mask = cv2.dilate(marker_mask,kernel,iterations=5)
            marker_edge = cv2.dilate(marker_mask,kernel,iterations=1) - marker_mask
            marker2_working[marker_mask.astype(bool)] = marker_oi
            marker2_working[marker_edge.astype(bool)] = -1
        return marker2_working

    def _generate_threshold(self,blur=None,threshold_mode=None):
        '''
        Method of creating threshold, uses different modes
        '''
        if threshold_mode == None:
            threshold_mode = self.threshold_mode
        
        if threshold_mode == "otsu":
            thresh = thresholding.otsu_threshold(self,blur)
            
        elif threshold_mode == "pixel":
            thresh = thresholding.pixel_threshold(self)
            
        elif threshold_mode == "local":
            thresh = thresholding.local_threshold(self)
        
        elif threshold_mode == "ensemble":
            threshold_mode = [thresholding.otsu_threshold,
                              thresholding.local_threshold,
                              thresholding.pixel_threshold
                              ]
            thresh = self._generate_threshold(threshold_mode=threshold_mode)

        elif isinstance(threshold_mode,list):
            tracker = np.full_like(self.image_cropped,0).astype(np.uint8)

            # Loop through all functions
            for thresh_func in threshold_mode:
                thresh_temp = thresh_func(self)
                print(np.shape(tracker))
                tracker = np.add(tracker,thresh_temp.astype(bool).astype(np.uint8))

            thresh = np.full_like(self.image_cropped,0).astype(np.uint8)
            thresh[tracker <= len(threshold_mode)/2] = 0 
            thresh[tracker > len(threshold_mode)/2] = 255

        elif callable(threshold_mode):
            thresh = threshold_mode(self)
            
        return thresh

    def _load_edge_classifier(self):
        '''
        Load the pixel classifer. Is a LARGE model, so only use this if needed
        '''
        if not self.edge_model:
            with open(STATIC_MODELS["edge_classifier"],"rb") as f:
                self.edge_model = pickle.load(f)

    def _edge_pixel_classifier(self):
        '''
        Experimental:
        Given a RandomForest pixel classifier, identify which edges are valid overlap or facet
        
        '''
        # Load model
        self._load_edge_classifier()

        # Featurize Image
        sigma_min = 1
        sigma_max = 32
        features_func = partial(feature.multiscale_basic_features,
                        intensity=False, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        channel_axis=None)
        features = features_func(self.image_cropped)

        # Flatten features
        (x,y,z) = features.shape
        features = features.reshape(x*y,z)

        # Predict
        results = future.predict_segmenter(features,self.edge_model)

        # Reshape
        thresh = 255*(results.reshape(x,y).astype(np.uint8)-1) # To make Overlap and Facet
        return thresh

    def _perform_edge_modification(self, edge_modification = None):
        '''
        Perform edge modification of threshold using internal schemes

        dark_bright: Use dark_bright edge detection scheme
        '''

        if not edge_modification:
            edge_modification = self.edge_modification
            #print(edge_modification)

        if edge_modification == None:
            img_edges = np.full_like(self.image_cropped,0).astype(np.uint8)
            return
        elif edge_modification == "canny":
            img_edges = em.edge_canny(self)

        elif edge_modification == "variance":
            img_edges = em.edge_variance(self)

        elif edge_modification == "darkbright": # Note: Probably add this as separate utility
            img_edges = em.edge_darkbright(self)

        elif edge_modification == "classifier":
            img_edges = em.edge_classifier(self)
            
        elif edge_modification == "localthresh":
            img_edges = em.edge_localthresh(self)

        elif edge_modification == "testing":
            img_edges = em.edge_testing(self)

        if img_edges is not None:
            self.thresh = self.thresh-img_edges
        else:
            # Assume something occurred in the function that
            # we wanted to happen
            return
        
    @property
    def img_edge(self):
        if self._img_edge is None:
            # This should be identical to "localthresh"
            # Get Masking info
            image_cropped_blur = cv2.GaussianBlur(self.image_cropped,(9,9),cv2.BORDER_DEFAULT)
            thresh = threshold_local(image_cropped_blur,35,offset=10)
            mask = self.image_cropped > thresh
            mask = em.quick_close(mask,3,neighbor_threshold=7)
            #mask = cv2.morphologyEx(mask.astype(np.uint8),)
            mask = cv2.dilate(mask.astype(np.uint8),np.ones([3,3]))
            mask = cv2.erode(mask.astype(np.uint8),np.ones([3,3]),iterations=2)
            img_logical = mask == 0
            #cv2.imwrite(f"localthresh{self._file_name}.png",img_logical.astype(np.uint8)*255)
            # Get Canny Edge
            canny_args = [(5,5),10,60,80,80,False]
            canny_edge = self.canny_edge(*canny_args)
            # This is used for "FACET SCORE"
            self._img_edge = copy.deepcopy(canny_edge)
        return self._img_edge

    def load_pixel_segmenter(self):
        '''
        Load the pixel classifer. Is a LARGE model, so only use this if needed
        '''
        if not self.pixel_model:
                with open(STATIC_MODELS["bg_segmenter"],"rb") as f:
                    self.pixel_model = pickle.load(f)

    def decorate_regions(self):
        '''
        Labels image 4 using information from Scikit to ensure commensurate labeling
        NOTE: The big issue is ensuring regions line up
        '''
        
        for i in self.regions_list:
            cx= int(self.outputs[3][i][0])
            cy= int(self.outputs[3][i][1])
            cv2.putText(self.image_labeled, text= str(i), org=(cx,cy),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0),
                    thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(self.image_labeled, (cx, cy), radius = 3, color=(0,0,0), thickness=-1)
        return


    @property
    def df(self): # Might not actually want this to be cached if we're making this interactive...
        if self._df is None:
            # Make sure to instantiate the images
            self.image_labeled

            file_present = os.path.isfile(self._csv_file)
            if file_present and not self.override_exists:
                df = pd.read_csv(self._csv_file)
                self.number_labels = len(df['area'])
                self._df = pd.read_csv(self._csv_file)
                self._region_tracker = self._df["Region"].min()
                return self._df

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
            clusters = measure.regionprops_table(self.markers2-self._label_increment, self.image_cropped,properties=propList)

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
            clusters['Labels'] = labeling_list
            clusters['Filename'] = filename_list
            clusters['Region'] = clusters['label']

            # Create df
            self._df = pd.DataFrame(clusters)
            self._region_tracker = self._df["Region"].min()

            # Need to add regional info
            if len(self.region_featurizers) > 0:
                region_dict = self.grab_region_dict(self.image_cropped,focused=False,alpha=0)
                def row_add_features(row:pd.Series):
                    region_img = region_dict[row.Region]
                    region = feat.Region(region_img,featurizers=self.region_featurizers)
                    return pd.Series(region.featurize())
                
                df_regions = self._df.apply(row_add_features, axis=1)
                self._df = pd.concat([self._df,df_regions], axis=1)
        

        return self._df

    def create_csv(self):
        if self.override_exists:
            self.df.to_csv(self._csv_file)
        else:
            print("WARNING: Override not in place")

    def to_h5(self,file_name,mode="w"):
        '''
        Save all images and regions to an h5 file for easy access
        Since some Regions may be skipped during measurement, need to key on this
        '''
        if file_name.split(".")[-1] != "h5":
            Exception(f"Error: {file_name} is not an h5 file. Change the extension")
        
        f = h5py.File(file_name,mode)

        group_name = Path(self._input_path).stem
        group = f.create_group(group_name)

        group.create_dataset("input_path",data=self.input_path)
        group.create_dataset("input_image",data=self.image_read)
        group.create_dataset("image_cropped",data=self.image_cropped)
        group.create_dataset("image_working",data=self.image_working)
        group.create_dataset("image_labeled",data=self.image_labeled)
        group.create_dataset("thresh",data=self.thresh)
        group.create_dataset("markers2",data=self.markers2)
       
        # Load in all regions identified
        #dset = group.create_dataset("Regions",shape=(len(self.df),*np.shape(self.markers2)))
        dset = group.create_dataset("Regions",shape=(self.df.Region.max(),*np.shape(self.markers2)))
        for ii,(key,region) in enumerate(self.grab_region_dict(self.image_cropped,focused=False,alpha=0).items()):
            # Some regions are grabbed in error
            if len(region) == 0:
                continue
            dset[key-1,:,:] = region # Subtract by 1 since regions are 1-indexing

        f.close()

    def _grab_region(self,img,region_oi,alpha=.75,buffer=20):
        label_oi = region_oi + self._label_increment
        mod_image = copy.deepcopy(img)
        label_marker = copy.deepcopy(self.markers2)
        label_marker[self.markers2 != label_oi] = 0

        y1 = grab_bound(label_marker,"top",buffer)
        y2 = grab_bound(label_marker,"bottom",buffer)
        x1 = grab_bound(label_marker,"left",buffer)
        x2 = grab_bound(label_marker,"right",buffer)

        mod_image[label_marker != label_oi] = mod_image[label_marker != label_oi]*alpha
        return mod_image[y1:y2,x1:x2]

    @property
    def region_arr(self):
        '''
        img Regions associated with the image segmentation 
        NOTE: _Slightly_ different from 'begin_labeling' as we remove non-region ENTIRELY
        '''
        if self._region_arr is None:
            self._region_arr = self.grab_region_array(focused=True)
        return self._region_arr

    @property
    def region_dict(self):
        if self._region_dict is None:

            self._region_dict = self.grab_region_dict(focused=True,alpha=.7)
        
        return self._region_dict
    
    def grab_region_array(self,img_oi=None,focused=True,alpha=0,buffer=5):
        '''
        Grab an array of images that are bounded (focused) or the same size as image_cropped (not focused)
        Can be useful for quickly making bools of regions
        '''
        if img_oi is None:
            img_oi = self.image_cropped
        self.df # Make sure this is initiated
        data_arr = []
        ii = 0
        regions_list = list(self.df["Region"])
        
        while ii < len(regions_list): # 1-Offset for counting purposes
            region_oi = regions_list[ii]
            if focused:
                data_arr.append(self._grab_region(img_oi,region_oi,alpha=alpha,buffer=buffer))
            if not focused:
                data_arr.append(self._grab_region(img_oi,region_oi,alpha=alpha,buffer=np.inf))
            ii += 1
        return data_arr
    
    def grab_region_dict(self,img_oi=None,focused=True,alpha=.7):

        if img_oi is None:
            img_oi = self.image_cropped
        self.df # Make sure this is initiated
        regions_list = list(self.df["Region"])
        data_dict = {}
        for region in regions_list: # 1-Offset for counting purposes
            region_oi = region
            if focused:
                data_dict[region] = self._grab_region(img_oi,region_oi,alpha=alpha,buffer=15)
            if not focused:
                data_dict[region] = self._grab_region(img_oi,region_oi,alpha=alpha,buffer=np.inf)
        return data_dict

    def begin_labeling(self,
                       labeling_dict={"C":"Crystal","M":"Multiple Crystal","P":"Poorly Segmented","I":"Incomplete"}):
        '''
        Major Utility function for labeling of segmented regions
        '''
        # Make sure B and D are not overwritten
        if "B" in labeling_dict or "D" in labeling_dict:
            raise Exception("Cannot use 'B' or 'D' in labeling_dict")
        
        # Develop options
        options_list = labeling_dict.keys()
        options_str = ", ".join([f'{key} = {val}' for key,val in labeling_dict.items()])

        
        self.df # To ensure it's been initialized
        ii = 0

        # NOTE: Use this instead of self.region_arr or self.region_dict to avoid overwrite issues
        regions_list = self.df["Region"]
        while ii < len(regions_list):
            clear_output(wait=False)
            region_oi = regions_list[ii] 

            testImage = self._grab_region(self.image_working,region_oi+self._label_increment,alpha=.75,buffer=20)
            plt.figure(figsize = (10,10))
            plt.imshow(testImage)
            plt.show()
            
            # User Input
            input_list = [*options_list,'B','D']
            
            print(f'Region {region_oi} (Max: {max(regions_list)}) \nNOTE: Skipping a region may mean a bad region was encountered\n')
            print("Type an integer to jump to region, or a character below to label image\n", 
                options_str,
                "\nB = Back, D = Done")
            user_input = input()
            while (user_input not in input_list) and (not user_input.isnumeric()):
                user_input = input("Invalid Input, Retry: ")
            if user_input == 'B':
                ii = ii - 1
                continue
            elif user_input.isnumeric():
                ii = int(user_input) - 1 # Because 1-Offset
                continue
            elif user_input == 'D':
                break
                
            # Clean-up
            translated_input = labeling_dict[user_input]

            self.df.loc[self.df['Region'] == region_oi,'Labels'] = translated_input
            self.df.to_csv(self._csv_file)
            
            ii = ii + 1

    ## PyQt Helper functions below
    def update_df_label_at_region(self,label,region=None):
        if region is None:
            region = self._region_tracker
        self.df.loc[self.df['Region'] == region,'Labels'] = label

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
            img_blur = cv2.GaussianBlur(self.image_cropped, blur_size,0)
            #img_blur = cv2.GaussianBlur(img_blur,blur_size,0)
        else:
            img_blur = cv2.bilateralFilter(self.image_cropped,d,ss,ss)
        self.edge = cv2.Canny(img_blur,tl,tu,apertureSize=3,L2gradient=True
                              )
        return self.edge

    
class BatchImageSegmenter():
    
    def __init__(self,
                img_list=None,
                IS_list=None,
                pixels_to_um=9.37,
                top_boundary=0,
                bottom_boundary=860,
                left_boundary=0,
                right_boundary=2560,
                result_folder_path="../../Results",
                override_exists=False,
                threshold_mode:Union[callable,str] = "otsu",
                edge_modification:Union[callable,str] = None,
                file_str = None):
        '''
        Class for doing batch processing of an image segmenter. 
        Compared to a regular ImageSegmenter, all functional and proeprty calls here simply grab
         and concatenate the individual ImageSegmenters together.

         Use this in cases where holding all images simultaneously is desirable, 
         but be warned it can take in a large amount of memory!
        '''
        self.pixels_to_um = pixels_to_um
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.result_folder_path = result_folder_path
        self.override_exists = override_exists
        self.threshold_mode = threshold_mode
        self.edge_modification = edge_modification
        self.file_str = file_str

        # Template
        self._template_IS = ImageSegmenter(
            pixels_to_um=self.pixels_to_um,
            top_boundary=self.top_boundary,
            bottom_boundary=self.bottom_boundary,
            left_boundary=self.left_boundary,
            right_boundary=self.right_boundary,
            result_folder_path=self.result_folder_path,
            override_exists=self.override_exists,
            threshold_mode=self.threshold_mode,
            edge_modification=self.edge_modification,
            file_str=self.file_str)

        # Input values
        self._img_list = None
        self._IS_list = None # Try to keep these updated in parallel w/ eachother

        if img_list:
            self.img_list = img_list
        if IS_list:
            self.IS_list = IS_list


        # pyqt helpers, will access the ImageSegmenters together
        self._df = None
        self._region_arr = None
        self._region_dict = None
        self._region_tracker = None

        self._batch_region_dict = None
        self._IS_index = None # Easiest way to check which ImageSegmenter we're in

        # Template ImageSegmenter (for if images are appended AFTER)
        
         
    @property
    def img_list(self):
        return self._img_list
    
    @img_list.setter
    def img_list(self,val):
        self._img_list = val
        
        self._IS_list = []
        for img_oi in self._img_list:
            ready_IS = copy.deepcopy(self._template_IS)
            ready_IS.input_path=img_oi
            self._IS_list.append(ready_IS)
    
    @property
    def IS_list(self):
        return self._IS_list

    @IS_list.setter
    def IS_list(self,val):
        self._IS_list = val

        self._img_list = []
        for IS in val:
            self._img_list.append(IS._input_path)
        

    def __getitem__(self,index):
        return self.IS_list[index]
    
    def __setitem__(self,index,newValue):
        if isinstance(newValue,ImageSegmenter):
            self._IS_list[index]=newValue
            self._img_list[index]=newValue.input_path
        else:
            self._img_list[index]=newValue
            ready_IS = copy.deepcopy(self._template_IS)
            ready_IS.input_path=newValue
            self._IS_list[index]=ready_IS

    def append(self,val):
        if isinstance(val,ImageSegmenter):
            self._IS_list.append(val)
            self._img_list.append(val.input_path)
        else:
            self._img_list.append(val)
            ready_IS = copy.deepcopy(self._template_IS)
            ready_IS.input_path=val
            self._IS_list.append(ready_IS)
            
    @property
    def df(self):
        '''
        Access the dataframe of EVERY ImageSegmenter here by concatting them
        '''
        self._df = pd.concat([IS.df for IS in self._IS_list])
        #if self._df is None:
        #    self._df = pd.concat([IS.df for IS in self._IS_list])
        return self._df

    @property
    def region_arr(self):
        if self._region_arr is None:
            self._region_arr = []
            for IS in self._IS_list:
                self._region_arr.extend(IS.region_arr)
        
        return self._region_arr
    
    @property
    def region_dict(self):
        if self._region_dict is None:
            self._batch_region_dict = BatchedRegionDict([IS.region_dict for IS in self._IS_list])
            self._region_dict = self._batch_region_dict
        
        return self._region_dict
    


# Need to define a batched region class to work
class BatchedRegionDict():
    def __init__(self,list_of_dicts):
        '''
        Dict-like class that only supports getting items
        Should make it easier to grab regions
        '''
        self.grouped_dict = {ii:dict_oi for ii,dict_oi in enumerate(list_of_dicts)}

    def __getitem__(self,val):
        val_tracker = val
        for key,item in self.grouped_dict.items():
            check_inside = val_tracker - len(item)
            
            if check_inside < 0:
                # Must be inside this current item
                return item[val_tracker]
            else:
                val_tracker = check_inside
    
        raise Exception("Exception: Out of range")

    def __setitem__(self,val):
        raise Exception("Setting values not supported")
    
    def __len__(self):
        return np.sum([len(ii) for _,ii in self.grouped_dict.items()])
        
    
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


def detectron2_maskrcnn_solids(img,folder_path=None):
    '''
    Use the trained Mask-RCNN in "Models"
    '''
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    if folder_path is None:
        folder_path = os.path.join(Path(__file__).parent.parent,"static","Models","detectron2")
    
    # Get cfg
    import yaml
    # with open(os.path.join(folder_path,"config.yaml"),"r") as f:
    #    cfg = yaml.load(f,Loader=yaml.BaseLoader)
    
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(folder_path,"config.yaml"))

    cfg.MODEL.WEIGHTS = os.path.join(folder_path,"model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

    predictor = DefaultPredictor(cfg)

    outputs = predictor(img)

    mask = outputs["instances"].pred_masks.to("cpu").numpy()
    '''
    z,y,x = np.shape(mask)
    for ii in np.arange(z):
        plt.imsave(f"mask{ii}.png",mask[ii,:,:])
    '''
    
    return mask
