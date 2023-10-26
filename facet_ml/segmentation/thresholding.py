import cv2
from functools import partial
from skimage import data, segmentation, feature, future
import numpy as np
from skimage.filters import threshold_local
from facet_ml.segmentation import edge_modification as em

## Design Note:
# Assume the function is passed an ImageSegmenter class and will always return a threshold of the image
def otsu_threshold(self,blur=None):
    '''
    Otsu binarization thresholding
    '''
    threshable = self.image_cropped if not blur else cv2.GaussianBlur(self.image_cropped, self.blur_size,0)
    ret, thresh = cv2.threshold(threshable, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Quick check
    #ret, thresh = cv2.threshold(threshable, ret-30, 255, cv2.THRESH_BINARY)
    return thresh

def pixel_threshold(self):
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
    features = features_func(self.image_cropped)

    # Flatten features
    (x,y,z) = features.shape
    features = features.reshape(x*y,z)

    # Predict
    results = future.predict_segmenter(features,self.pixel_model)

    # Reshape
    thresh = 255*(results.reshape(x,y).astype(np.uint8)-1) # To make background and not
    return thresh

def local_threshold(self):
    '''
    Use Adaptive (local) thresholding
    '''
    image_cropped_blur = cv2.GaussianBlur(self.image_cropped,(9,9),cv2.BORDER_DEFAULT)
    thresh = cv2.adaptiveThreshold(self.image_cropped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,53,10)
    tl = (image_cropped_blur > threshold_local(image_cropped_blur,35,offset=10)).astype(np.uint8)*255
    tl = em.quick_close(tl,3,3)
    return tl