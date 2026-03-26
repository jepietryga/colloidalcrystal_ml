from facet_ml.segmentation import segmenter
from facet_ml.static.path import *
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import os
import pytest

test_path = Path(__file__).parent / "4 nM 1.bmp"
test_image = cv2.imread(str(test_path), 0)

def test_segmenter_AlgorithmicSegmenter_noModels():

    for th_mode in ["otsu","local"]:
        for em_mode in [None,"localthresh"]:
            AS = segmenter.AlgorithmicSegmenter(test_image,
                                                th_mode,
                                                em_mode)

            im = AS.markers_filled

            plt.imshow(im)
            # plt.savefig(f"{th_mode}_{em_mode}.png")

def test_segmenter_AlgorithmicSegmenter_Models():

    for th_mode in ["pixel","ensemble"]:
        for em_mode in [None,"localthresh"]:
            AS = segmenter.AlgorithmicSegmenter(test_image,
                                                th_mode,
                                                em_mode)

            im = AS.markers_filled

            plt.imshow(im)
            # plt.savefig(f"{th_mode}_{em_mode}.png")
    
def test_segmenter_MaskRCNNSegmenter():

    # Check if the model is available to use. If it isn't, this test should fail automatically.
    if not os.path.exists(STATIC_MODELS["maskrcnn"]):
        pytest.skip("Mask R-CNN model file is not available")
    
    MS = segmenter.MaskRCNNSegmenter(test_image,device="cpu")
    im = MS.markers_filled
    plt.imshow(im)
    # plt.savefig("MaskRCNNSegmenter.png")
    #raise NotImplemented

def test_segmenter_SAMSegmenter():
    if not os.path.exists(STATIC_MODELS["segment_anything_vit_l"]):
        pytest.skip("Segment Anything model file is not available")
    SS = segmenter.SAMSegmenter(test_image,device="cpu")
    im = SS.markers_filled
    plt.imshow(im)
    # plt.savefig("SamSegmenter.png")
    #raise NotImplemented

def test_default_ImageSegmenter():
    IS = segmenter.ImageSegmenter(input_path=test_image)
    #IS = segmenter.ImageSegmenter(input_path=str(test_path))
    output_path = Path("test.h5")
    try:
        IS.to_h5(output_path)
    finally:
        if output_path.exists():
            output_path.unlink()

if __name__ == "__main__":
    test_segmenter_AlgorithmicSegmenter()
    test_segmenter_MaskRCNNSegmenter()
    test_segmenter_SAMSegmenter()
    test_default_ImageSegmenter()
