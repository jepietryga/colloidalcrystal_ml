from facet_ml.segmentation import segmenter
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

test_path = Path(__file__).parent.parent / "Images" / "Training" / "4 nM 1.bmp"
test_image = cv2.imread(str(test_path),0)
cv2.imwrite("tmp_test.png",test_image)

def test_segmenter_AlgorithmicSegmenter():

    for th_mode in ["otsu","local","pixel","ensemble"]:
        for em_mode in [None,"localthresh"]:
            AS = segmenter.AlgorithmicSegmenter(test_image,
                                                th_mode,
                                                em_mode)

            im = AS.markers_filled

            plt.imshow(im)
            plt.savefig(f"{th_mode}_{em_mode}.png")
    
def test_segmenter_MaskRCNNSegmenter():
    MS = segmenter.MaskRCNNSegmenter(test_image)
    im = MS.markers_filled
    plt.imshow(im)
    plt.savefig("MaskRCNNSegmenter.png")
    #raise NotImplemented

def test_segmenter_SAMSegmenter():
    SS = segmenter.SAMSegmenter(test_image)
    im = SS.markers_filled
    plt.imshow(im)
    plt.savefig("SamSegmenter.png")
    #raise NotImplemented

def test_default_ImageSegmenter():
    IS = segmenter.ImageSegmenter(input_path=test_image)
    #IS = segmenter.ImageSegmenter(input_path=str(test_path))
    IS.to_h5("test.h5")

if __name__ == "__main__":
    test_segmenter_AlgorithmicSegmenter()
    test_segmenter_MaskRCNNSegmenter()
    test_segmenter_SAMSegmenter()
    test_default_ImageSegmenter()