from PyQt5 import QtWidgets, uic,QtCore,QtGui,QtWidgets
from PyQt5.QtGui import qRgb
import sys
sys.path.append("../..")
import cv2
import numpy as np
from facet_ml.segmentation.segmenter import ImageSegmenter


threshold_mode_mapper = {
    "Otsu (Global) Binarization":"otsu",
    "Local Threshold":"local",
    "Pixel Classifier":"pixel",
    "Ensemble":"ensemble",
    "Detectron2":"detectron2"
}

edge_mode_mapper = {
    "Local Thresholding":"localthresh",
    "Bright-Dark":"darkbright",
    "None":None,
    "Testing":"testing"
}

QImage = QtGui.QImage

def toQimage(im,copy=False):
    if im is None:
        return QImage()
    gray_color_table = [qRgb(i, i, i) for i in range(256)]
    if im.dtype == np.uint8:
        if len(im.shape) == 2:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                return qim.copy() if copy else qim

    raise NotImplemented

def cv_to_QPixMap(cv_img,copy=False):
    pix_Q = QtGui.QPixmap(toQimage(cv_img.astype(np.uint8)))
    return pix_Q
    '''
    height, width = (cv_img.shape[0],cv_img.shape[1])
    bytesPerLine = 3 * width
    qImg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    pix_Q = QtGui.QPixmap(qImg)
    return pix_Q
    '''


class Ui(QtWidgets.QDialog):
#class Ui(QtWidgets):
    def __init__(self):
        super(Ui,self).__init__()
        uic.loadUi('segmenter_classify_train_v2.ui',self)

        # Load all necessary connections
        self.wire_button()
        self.show()

        # Check cv images
        self.image_segmenter = None
        self.cv_img = None
        self.file_path = None


    def wire_button(self):
        self.selectInputPushButton.clicked.connect(self.get_input_file)
        self.runSegmentationPushButton.clicked.connect(self.perform_segmentation)

    def get_input_file(self):
        fileName,_ = QtWidgets.QFileDialog.getOpenFileName(self,'Single File',"./",filter="Images (*.bmp *.png *.tif)")
        self.file_path = fileName
        
        self.perform_segmentation()
        self.labeling_mode()
    
    def perform_segmentation(self):
        threshold_mode = self.thresholdMethodComboBox.currentText()
        edge_mode = self.edgeMethodMethodComboBox.currentText()


        self.image_segmenter = ImageSegmenter(self.file_path,
                                              threshold_mode=threshold_mode_mapper[threshold_mode],
                                              edge_modification=edge_mode_mapper[edge_mode])

        input_pix = cv_to_QPixMap(self.image_segmenter.image_read)
        thresh_pix = cv_to_QPixMap(self.image_segmenter.thresh)
        markers_pix = cv_to_QPixMap(self.image_segmenter.markers2)

        self.inputImagePixLabel.setPixmap(input_pix)
        self.inputImagePixLabel.setScaledContents(True)
        self.thresholdImagePixLabel.setPixmap(thresh_pix)
        self.thresholdImagePixLabel.setScaledContents(True)
        self.markersImagePixLabel.setPixmap(markers_pix)
        self.markersImagePixLabel.setScaledContents(True)
        
    def labeling_mode(self):
        self.image_segmenter

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()