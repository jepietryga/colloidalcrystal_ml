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
    "Detectron2":"detectron2",
    "Segment Anything":"segment_anything"
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

        # Check cv images
        self.batch_image_segmenter = None
        self.batch_tracker = None
        self.image_segmenter = None
        self.cv_img = None
        self.file_path = None

        # Load all necessary connections
        # self.wire_functionality()
        self.wire_core_buttons()
        self.wire_classify_buttons()
        self.show()

        
    def wire_core_buttons(self):
        self.selectInputPushButton.clicked.connect(self.get_input_file)
        self.runSegmentationPushButton.clicked.connect(self.perform_segmentation)
        self.saveSegmentationButton.clicked.connect(self.save_segmentation)

    def wire_classify_buttons(self):

        self.forwardClassifyButton.clicked.connect(self.forward_classify_click)
        self.backClassifyButton.clicked.connect(self.back_classify_click)
        self.labelingGuideLineEdit.editingFinished.connect(self.update_df_region_label)
        self.saveClassifyButton.clicked.connect(self.save_classify)

    ## Wired Functions

    def get_input_file(self):
        
        fileName,_ = QtWidgets.QFileDialog.getOpenFileName(self,'Single File',"./",filter="Images (*.bmp *.png *.tif)")
        
        
        if fileName:
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
        
        # Update state of PyQt
        self.refresh_classify_tab()

    def save_classify(self):
        if self.image_segmenter:
            file_name,_ = QtWidgets.QFileDialog.getSaveFileName(self,"Save File","","CSV Files (*.csv)")
            if file_name:
                self.image_segmenter.df.to_csv(file_name)
    
    def save_segmentation(self):
        if self.image_segmenter:
            file_name,_ = QtWidgets.QFileDialog.getSaveFileName(self,"Save File","","H5 Files (*.h5)")
            if file_name:
                self.image_segmenter.to_h5(file_name)

    def back_classify_click(self):
        if self.image_segmenter:
            self.move_region(-1)
            self.refresh_classify_tab(self)
    
    def forward_classify_click(self):
        if self.image_segmenter:
            self.move_region(1)
            self.refresh_classify_tab()

    def update_df_region_label(self):
        text = self.labelingGuideLineEdit.text()
        self.image_segmenter.update_df_label_at_region(text)
        # Note: Might remove this
        self.forward_classify_click()

    ## Utility functions
    def move_region(self,val):
        '''
        Need to check if the region actually exists as some may disappear if neglibible size
        '''
        check_val = 0 
        for ii in range(len(self.image_segmenter.df)):
            check_val += val
            region_oi = check_val+self.image_segmenter._region_tracker
            rolled_region = self.region_roll(region_oi)
            region_valid = rolled_region in self.image_segmenter.df["Region"].to_list()
            if region_valid:
                self.image_segmenter._region_tracker = rolled_region
                return
       
        raise Exception("No valid regions found")

    def region_roll(self,val):
        if val > self.image_segmenter.df["Region"].max():
            return self.image_segmenter.df["Region"].min()
        elif val < self.image_segmenter.df["Region"].min():
            return self.image_segmenter.df["Region"].max()
        else:
            return val

    def _region_roll(self,val):
        #print(self.image_segmenter.df["Region"])
        val
        print(self.image_segmenter.df["Region"].max(),type(self.image_segmenter.df["Region"].max()))
        if self.image_segmenter._region_tracker > self.image_segmenter.df["Region"].max():
            self.image_segmenter._region_tracker = self.image_segmenter.df["Region"].min()
        elif self.image_segmenter._region_tracker < self.image_segmenter.df["Region"].min():
            self.image_segmenter._region_tracker = self.image_segmenter.df["Region"].max()
        else:
            return val

    ## Large Scale Functions

    def refresh_classify_tab(self,_=None):
        '''
        Given current state of image segmenter, update the PyQt app to display image, info, and label
        '''
        print(self.image_segmenter._region_tracker)
        # Image
        region_pix = cv_to_QPixMap(self.image_segmenter.region_dict[self.image_segmenter._region_tracker])
        self.classifyRegionPixLabel.setPixmap(region_pix)
        self.classifyRegionPixLabel.setScaledContents(True)

        # Update Text
        guiding_text = f"Looking at Region {self.image_segmenter._region_tracker} (Total Regions: {len(self.image_segmenter.df)})"
        guiding_text += "\nWrite label for region in field below and hit Enter"
        self.labelingGuideBodyLabel.setText(guiding_text)

        # Update Label
        label_text = self.image_segmenter.df.loc[
            self.image_segmenter.df["Region"] == self.image_segmenter._region_tracker,"Labels"
        ].tolist()[0]
        self.labelingGuideLineEdit.setText(label_text)

    def initiate_labeling(self):
        '''
        Classify's Tab's labeling begins through this function
        '''
        # Wire the Back, Forward buttons

        cv_to_QPixMap(self.image_segmenter.region_dict[self.image_segmenter._region_tracker])
     
    def labeling_mode(self):
        self.image_segmenter

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()