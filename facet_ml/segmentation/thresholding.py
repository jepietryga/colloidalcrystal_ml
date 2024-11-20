import cv2
from functools import partial
from skimage import feature, future
import numpy as np
from skimage.filters import threshold_local
from facet_ml.segmentation import edge_modification as em


## Design Note:
# Assume the function is passed an Segmenter class and will always return a threshold of the image
def otsu_threshold(segmenter, blur=None):
    """
    Otsu binarization thresholding
    """
    threshable = (
        segmenter.image if not blur else cv2.GaussianBlur(segmenter.image, segmenter.blur_size, 0)
    )
    ret, thresh = cv2.threshold(threshable, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Quick check
    # ret, thresh = cv2.threshold(threshable, ret-30, 255, cv2.THRESH_BINARY)
    return thresh


def pixel_threshold(segmenter):
    """
    Use Weka-inspired pixel segmenter to generate threshold
    """
    # Load model
    segmenter.load_pixel_segmenter()

    # Featurize Image
    sigma_min = 1
    sigma_max = 16
    features_func = partial(
        feature.multiscale_basic_features,
        intensity=False,
        edges=True,
        texture=True,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        channel_axis=None,
    )
    features = features_func(segmenter.image)

    # Flatten features
    (x, y, z) = features.shape
    features = features.reshape(x * y, z)

    # Predict
    results = future.predict_segmenter(features, segmenter.pixel_model)

    # Reshape
    thresh = 255 * (
        results.reshape(x, y).astype(np.uint8) - 1
    )  # To make background and not
    return thresh


def local_threshold(segmenter):
    """
    Use Adaptive (local) thresholding
    """
    image_blur = cv2.GaussianBlur(segmenter.image, (9, 9), cv2.BORDER_DEFAULT)
    thresh = cv2.adaptiveThreshold(
        segmenter.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 53, 10
    )
    tl = (image_blur > threshold_local(image_blur, 35, offset=10)).astype(
        np.uint8
    ) * 255
    tl = em.quick_close(tl, 3, 3)
    return tl


def ensemble_threshold(segmenter):
    '''
    Combined multiple thresholdd methods (otsu, local, and pixel)
    '''
    threshold_list = [
        otsu_threshold,
        local_threshold,
        pixel_threshold,
    ]
    thresh = multi_threshold(segmenter, threshold_list)
    return thresh


def multi_threshold(segmenter, threshold_callables):
    """
    Given list of callables, run each and ensemble decision-make
    on using majority decision on each callable's result
    Args:
        segmenter (AlgorithmicSegmenter) : Segmenter class with a held image
        threshold_callables (list[callables]) : List of fucntions tha accept just segmenter
    """
    tracker = np.full_like(segmenter.image, 0).astype(np.uint8)

    # Loop through all functions
    for thresh_func in threshold_callables:
        thresh_temp = thresh_func(segmenter)
        tracker = np.add(tracker, thresh_temp.astype(bool).astype(np.uint8))

    thresh = np.full_like(segmenter.image, 0).astype(np.uint8)
    thresh[tracker <= len(threshold_callables) / 2] = 0
    thresh[tracker > len(threshold_callables) / 2] = 255

    return thresh
