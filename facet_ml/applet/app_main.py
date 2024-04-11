from PyQt5 import QtWidgets, uic,QtCore,QtGui,QtWidgets
from PyQt5.QtGui import qRgb
import sys
sys.path.append("../..")
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from facet_ml.segmentation.segmenter import BatchImageSegmenter,ImageSegmenter
from facet_ml.segmentation.segmenter import (
    AlgorithmicSegmenter,
    MaskRCNNSegmenter,
    SAMSegmenter
)
import h5py

segment_mode_mapper = {
    "Otsu (Global) Binarization":{"segmenter":AlgorithmicSegmenter,
                                  "segmenter_kwargs":{"threshold_mode":"otsu"},
                                },
    "Local Threshold":{"segmenter":AlgorithmicSegmenter,
                                  "segmenter_kwargs":{"threshold_mode":"localthresh"},
                                },
    "Pixel Classifier":{"segmenter":AlgorithmicSegmenter,
                                  "segmenter_kwargs":{"threshold_mode":"pixel"},
                                },
    "Ensemble":{"segmenter":AlgorithmicSegmenter,
                                  "segmenter_kwargs":{"threshold_mode":"ensemble"},
                                },
    "Detectron2":{"segmenter":MaskRCNNSegmenter,
                                  "segmenter_kwargs":{},
                                },
    "Segment Anything":{"segmenter":SAMSegmenter,
                                  "segmenter_kwargs":{"points_per_side":64},
                                },
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
        path = Path(__file__).parent / 'segmenter_classify_train_v2.ui'
        uic.loadUi( path,self)

        # Check cv images
        self.batch_image_segmenter = None
        self.batch_tracker = None
        self.image_segmenter = None
        self.cv_img = None
        self.file_path = None
        
        # Image Reading Variables
        self.top_boundary = None
        self.right_boundary = None
        self.bottom_boundary = None
        self.left_boundary = None
        self.pixels_to_um = None

        # Load all necessary connections
        # self.wire_functionality()
        self.wire_core_buttons()
        self.wire_input_image_buttons()
        self.wire_classify_buttons()
        self.inputImagePixLabel.installEventFilter(self)

        # Show the window
        self.show()    

    def wire_core_buttons(self):
        self.selectInputPushButton.clicked.connect(self.get_input_file)
        self.runSegmentationPushButton.clicked.connect(self.perform_segmentation)
        self.saveSegmentationButton.clicked.connect(self.save_segmentation)

    def wire_input_image_buttons(self):
        self.inputRightPushButton.clicked.connect(self.input_right)
        self.inputLeftPushButton.clicked.connect(self.input_left)

        # Line Edits
        self.topBoundInputLineEdit.editingFinished.connect(self.update_image_reading)
        self.rightBoundInputLineEdit.editingFinished.connect(self.update_image_reading)
        self.bottomBoundInputLineEdit.editingFinished.connect(self.update_image_reading)
        self.leftBoundInputLineEdit.editingFinished.connect(self.update_image_reading)
        self.scaleInputLineEdit.editingFinished.connect(self.update_image_reading)



    def wire_classify_buttons(self):

        self.forwardClassifyButton.clicked.connect(self.forward_classify_click)
        self.backClassifyButton.clicked.connect(self.back_classify_click)
        self.labelingGuideLineEdit.editingFinished.connect(self.update_df_region_label)
        self.saveClassifyButton.clicked.connect(self.save_classify)

    ## Wired Functions

    def get_input_file(self):
        
        filter = "Images (*.bmp *.png *.tif)"
    
        fileDialog = QtWidgets.QFileDialog()
        fileDialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        names,_ = fileDialog.getOpenFileNames(self, "Open files", "./", filter)
        #print(names)
        
        #fileName,_ = QtWidgets.QFileDialog.getOpenFileName(self,'Single File',"./",filter=filter)
        
        
        if names:
            self.files_list = names
            #self.file_path = fileName
            
            self.perform_segmentation()
            #self.labeling_mode()

    def input_right(self):
        self.batch_tracker += 1
        self.check_input_button_enable()
        self.update_image_segmenter()
    
    def input_left(self):
        self.batch_tracker -= 1
        self.check_input_button_enable()
        self.update_image_segmenter()

    def perform_segmentation(self):
        segment_mode = self.segmentMethodComboBox.currentText()
        edge_mode = self.edgeMethodMethodComboBox.currentText()

        # Re-runs everything and updates visible image_segmenter
        self.update_image_reading()

        segmenter_mode = segment_mode_mapper[segment_mode]
        edge_modification = edge_mode_mapper[edge_mode]
        if isinstance(segmenter_mode["segmenter"],AlgorithmicSegmenter):
            segmenter_mode["segmenter_kwargs"] = segmenter_mode["segmenter_kwargs"] | {"edge_modification":edge_modification}
        self.batch_image_segmenter = BatchImageSegmenter(self.files_list,
                                            **segmenter_mode,
                                            top_boundary=self.top_boundary,
                                            bottom_boundary=self.bottom_boundary,
                                            right_boundary=self.right_boundary,
                                            left_boundary=self.left_boundary,
                                            pixels_to_um=self.pixels_to_um
                                            )
        if self.batch_tracker is None:
            self.batch_tracker=0
        
        self.batch_process()
        self.update_image_segmenter() # NOTE: Also updates tabs
        self.check_input_button_enable()

    def save_classify(self):
        if self.image_segmenter:
            group_name = Path(self.image_segmenter._input_path).stem
            
            file_name,_ = QtWidgets.QFileDialog.getSaveFileName(self,"Save File",f"{group_name}.csv","CSV Files (*.csv)")
            if file_name:
                self.image_segmenter.df.to_csv(file_name)
    
    def save_segmentation(self):
        if self.image_segmenter:
            group_name = Path(self.image_segmenter._input_path).stem
            
            file_name,_ = QtWidgets.QFileDialog.getSaveFileName(self,"Save File",f"{group_name}.h5","H5 Files (*.h5)")
            if file_name:
                f = h5py.File(file_name,"w")
                f.close()
                for IS in self.batch_image_segmenter.IS_list:
                    IS.to_h5(file_name,"r+")

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

    def update_image_reading(self):
        self.top_boundary = int(self.topBoundInputLineEdit.text())
        self.right_boundary = int(self.rightBoundInputLineEdit.text())
        self.bottom_boundary = int(self.bottomBoundInputLineEdit.text())
        self.left_boundary = int(self.leftBoundInputLineEdit.text())
        self.pixels_to_um = float(self.scaleInputLineEdit.text())

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

    def check_input_button_enable(self):
        n_files = len(self.files_list)
        if self.batch_tracker <= 0:
            left_bool = False
        else:
            left_bool = True

        if self.batch_tracker >= n_files-1:
            right_bool = False
        else:
            right_bool = True

        self.inputLeftPushButton.setEnabled(left_bool)
        self.inputRightPushButton.setEnabled(right_bool)

    def enable_classify_buttons(self,val:bool):
        self.forwardClassifyButton.setEnabled(val)
        self.backClassifyButton.setEnabled(val)
        
    ## Large Scale Functions

    def batch_process(self):
        self.pbar = QtWidgets.QProgressBar(self)
        self.pbar.setGeometry(80, 205, 240, 30)
        self.pbar.setRange(0,len(self.files_list))
        self.pbar.show()
        self.show()
        print("PROGRESS WINDOW OPEN")
        for ii,IS in enumerate(self.batch_image_segmenter._IS_list):
            self.pbar.setValue(ii)
            print(ii)
            QtWidgets.QApplication.processEvents() 
            IS.process_images()
        print("PROGRESS WINDOW CLOSE")
        self.pbar.hide()
        self.pbar.close()

    def update_image_segmenter(self):
        self.image_segmenter=self.batch_image_segmenter[self.batch_tracker]
        self.refresh_segmenter_tab()
        self.refresh_classify_tab()

    def refresh_segmenter_tab(self):
        input_pix = cv_to_QPixMap(self.image_segmenter.image_read)
        thresh_pix = cv_to_QPixMap(self.image_segmenter.thresh)
        markers_pix = cv_to_QPixMap(self.image_segmenter.markers_filled)

        self.inputImagePixLabel.setPixmap(input_pix)
        self.inputImagePixLabel.setScaledContents(True)
        self.segmentImagePixLabel.setPixmap(thresh_pix)
        self.segmentImagePixLabel.setScaledContents(True)
        self.markersImagePixLabel.setPixmap(markers_pix)
        self.markersImagePixLabel.setScaledContents(True)
        

    def refresh_classify_tab(self,_=None):
        '''
        Given current state of image segmenter, update the PyQt app to display image, info, and label
        '''
        #print(self.image_segmenter._region_tracker)
        self.image_segmenter.df # Ensure this is instantiated
        if ~np.isnan(self.image_segmenter._region_tracker):
            self.enable_classify_buttons(True)
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
            #print(label_text)
            self.labelingGuideLineEdit.setText(str(label_text))
        else:
            self.enable_classify_buttons(False)
            blank_pix = QtGui.QPixmap()
            blank_pix.fill(QtGui.QColor(0,0,0,0))
            self.classifyRegionPixLabel.setPixmap(blank_pix)
            self.classifyRegionPixLabel.setScaledContents(True)

            # Update Text
            guiding_text = "No regions detected in this image"
            self.labelingGuideBodyLabel.setText(guiding_text)
            

    def initiate_labeling(self):
        '''
        Classify's Tab's labeling begins through this function
        '''
        # Wire the Back, Forward buttons

        cv_to_QPixMap(self.image_segmenter.region_dict[self.image_segmenter._region_tracker])


    ## Drag and drop functionality
    def eventFilter(self, object, event):
        if (object is self.inputImagePixLabel):
            if (event.type() == QtCore.QEvent.DragEnter):
                if event.mimeData().hasUrls():
                    event.accept()   # must accept the dragEnterEvent or else the dropEvent can't occur !!!
                    
                else:
                    event.ignore()
                    
            if (event.type() == QtCore.QEvent.Drop):
                if event.mimeData().hasUrls():   # if file or link is dropped
                    urlcount = len(event.mimeData().urls())  # count number of drops
                    paths = []
                    ext_list = [".tif",".bmp",".jpg",".jpeg",".png"]
                    for Qurl in event.mimeData().urls():
                        url = Qurl.toLocalFile()
                        # If it's a folder...
                        if os.path.isdir(url):
                            paths.extend([p for p in glob.glob(os.path.join(url,"*"))
                                               if any([(check in p) for check in ext_list]) 
                                               ]
                            )
                        # If they're images...
                        else:
                            if any([(check in url) for check in ext_list]):
                                paths.append(url)
                    self.files_list = paths
                    self.perform_segmentation()
                    #event.accept()  # doesnt appear to be needed
            return False # lets the event continue to the edit
        return False


def run_app():
    '''
    Function to run applet
    '''
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()

if __name__ == "__main__":
    run_app()