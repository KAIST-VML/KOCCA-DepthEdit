from cProfile import label
import sys,os
from datetime import datetime
import time 

from functools import partial
import time
from matplotlib import widgets
import matplotlib.pyplot as plt

import numpy as np
import cv2
import glob
import qimage2ndarray

from PyQt5 import uic
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QStringListModel,Qt
import kornia

from options.test_options import TestOptions
from infer import infer

import resource_rc
import argparse
from collections import deque

def flood_fill(img, out, x_0, y_0, color, threshold=0):
    h,w,c = img.shape
    def in_boundary(x,y,h,w):
        if x >= 0 and x < w and y >= 0 and y < h:
            return True
        else:
            return False

    source_value = img[y_0,x_0].copy()
    target_value = np.array([color.red(),color.green(),color.blue(),255])
  
    visited = set()
    visited.add((x_0,y_0))
    candidate = deque([(x_0,y_0)])

    out[y_0,x_0] = target_value
    while len(candidate) > 0:
        x,y = candidate.popleft()
        for (xx, yy) in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if not (xx,yy) in visited and in_boundary(xx,yy,h,w):
                visited.add((xx,yy)) 
                if (abs(img[yy,xx].astype(np.int16)-source_value.astype(np.int16)) <= threshold).all():
                        out[yy,xx] = target_value
                        candidate.append((xx,yy))
                    
    return out

class Form(QMainWindow):
    displaySize = 300
    pickerActive = False
    '''
    4 MODE for Inference
    1) Depth alone
    2) Depth + Segmentation Guide
    3) Depth + Segmentation Guide (pure SPADE)
    4) Refinement
    '''
    fileName = None
    img = None
    disp = None

    modeType = {"Depth":"depth", "Semantic Guided Depth":"segDepth", "Segmap":"segMap", "Refine":"refine"}
    mode = "depth"
    model = None
    labelModel = None
    labelImage = None
    params = {}
    figureWidget = None
    
    max_history = 3
    prev_guides=deque(maxlen=max_history)
    userstudyTask = None

    def __init__(self):
        super(Form, self).__init__()
        self.opt = TestOptions().parse(); self.opt.lambda_feat = 1.0; self.opt.lambda_seg = 1.0

        self.dirName = os.path.dirname(__file__)
        self.ui = uic.loadUi("./mainwindow_aligned2.ui", self)
        self.ui.setWindowTitle("Interactive Single Image Depth Extractor")
        
        #connect ui objects with functions
        self.ui.loadButton.clicked.connect(lambda: self.loadImage('image'))
        self.ui.processButton.clicked.connect(self.processImage)
        self.ui.strokeLoadBtn.clicked.connect(lambda: self.loadImage('stroke'))
        self.ui.saveButton.clicked.connect(lambda: self.saveImage('depth'))
        self.ui.plotButton.clicked.connect(self.plotImage)
        self.ui.strokeSaveBtn.clicked.connect(lambda: self.saveImage("stroke"))
        self.ui.valueSlider.valueChanged.connect(self.changeValue)
        self.ui.sizeSlider.valueChanged.connect(self.changeSize)
        self.ui.clearButton.clicked.connect(self.eraseAll)
        self.ui.strokeButton.clicked.connect(self.activateStroke)
        self.ui.eraseButton.clicked.connect(self.activateEraser)
        self.ui.fillButton.clicked.connect(self.activateFill)
        self.ui.ckptCB.currentTextChanged.connect(self.refreshModel)
        self.ui.optCB.currentTextChanged.connect(self.updateOptionLE)
        self.ui.optLE.returnPressed.connect(self.updateOption)
        self.ui.searchLE.returnPressed.connect(self.updateOptionCB)
        self.ui.showStrokeCB.stateChanged.connect(self.toggleGuideImage)
        self.ui.showDepthCB.stateChanged.connect(self.toggleDepthImage)
        self.ui.showColormapCB.stateChanged.connect(self.toggleColormap)
        self.ui.showOverlayCB.stateChanged.connect(self.toggleOverlay)


        self.ui.valueBox.mouseReleaseEvent = partial(self.activatePicker)

        # Key Shortcuts
        self.undo_SC = QShortcut(QtGui.QKeySequence('Ctrl+Z'), self)
        self.undo_SC.activated.connect(self.undoStroke)
        
        # Size options
        self.opt.infer_size = 384
        self.opt.fit_max = False

        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')
        self.pen_size = int(self.ui.sizeLabel.text())
        self.initialize()

        self.ui.show()

        self.params['option'] = self.opt
        self.processTask = TaskThread(params=self.params)
        self.processTask.status = 'launch'
        self.processTask.timeSignal.connect(self.initProgress)
        self.processTask.start()

        # self.processTask.notifyProgress.connect(self.onProgress)
    
    def initialize(self):
        self.msg = QMessageBox()

        self.ui.progressBar.setValue(0)
        self.ui.statusBar.showMessage("Status: Load Image for Depth Extraction")
        self.ui.valueSlider.setRange(0,100)
        self.ui.valueSlider.setSingleStep(10)
        self.ui.sizeSlider.setRange(2,20)
        self.ui.sizeSlider.setSingleStep(1)

        self.currentColor = self.ui.valueSlider.value() / 100.0 * 255
        length = self.ui.valueBox.width()
        # self.colorPixmap = QtGui.QPixmap(length,length)
        # self.colorPixmap.fill(QtGui.QColor(self.currentColor,self.currentColor,self.currentColor))
        # self.ui.valueBox.setPixmap(self.colorPixmap)
        self.ui.valueBox.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        
        self.brushPixmap = QtGui.QPixmap(self.pen_size,self.pen_size)
        currentColor = int(self.currentColor*255)
        self.brushPixmap.fill(QtGui.QColor(currentColor,currentColor,currentColor))
        self.ui.valueBox.setPixmap(self.brushPixmap)

        #initialization for graphics
        self.ui.imageDisplay.setStyleSheet('QLabel  {background-color: #2b2b2b;}')
        self.ui.imageBox.setStyleSheet('QGroupBox   {background-color: #2b2b2b;}')
        self.ui.depthDisplay.setStyleSheet('QLabel  {background-color: #2b2b2b;}')
        self.ui.depthBox.setStyleSheet('QGroupBox   {background-color: #2b2b2b;}')

        self.displaySize = (self.ui.imageDisplay.size().width(),
                            self.ui.imageDisplay.size().height())
        self.displayBoxSize = (self.ui.imageBox.size().width(),
                            self.ui.imageBox.size().height())

        self.updateMode()
        self.initializeOption()

        #initialize segmentation modules
        self.label_classes = np.load("label.npy", allow_pickle=True).item()
        self.label_classes['unknown'] = 255 

    def resizeEvent(self, event):
        self.displayBoxSize = (self.ui.imageBox.size().width(),
                            self.ui.imageBox.size().height())
        self.displaySize = (self.displayBoxSize[0]-10, self.displayBoxSize[1]-10)
        w, h = self.displaySize
        self.ui.imageDisplay.resize(w,h)
        self.ui.depthDisplay.resize(w,h)

        self.resizeImages()
        QtWidgets.QMainWindow.resizeEvent(self, event)

    def resizeImages(self):
        def resizeByScale(w,h,ww,hh,img):
            ratio1, ratio2 = w/h, ww/hh
            if ratio1 > ratio2:
                return img.scaledToHeight(h)
            else:
                return img.scaledToWidth(w)
        
        if self.fileName is None:
            return

        w, h = self.displaySize
        hh, ww = self.img.shape[0], self.img.shape[1]
        
        if self.fileName is not None:
            img = QtGui.QImage(self.fileName)
            img2 = resizeByScale(w,h,ww,hh,img)
            self.image = QtGui.QPixmap.fromImage(img2)
            if self.guide is not None:
                self.guide = resizeByScale(w,h,ww,hh,self.guide)
                results = self.join_pixmap(self.image, self.guide)
                self.ui.imageDisplay.setPixmap(results)
            else:
                self.ui.imageDisplay.setPixmap(self.image)

        if self.disp is not None:
            viewDisp = self.ui.depthDisplay.pixmap()
            viewDisp2 = resizeByScale(w,h,ww,hh,viewDisp)
            self.ui.depthDisplay.setPixmap(viewDisp2)

    def initializeOption(self):
        import inspect
        attributes = inspect.getmembers(self.opt, lambda a:not(inspect.isroutine(a)))
        self.optionDict = {}
        for a in attributes:
            if not (a[0].startswith('__') and a[0].endswith('__')):
                self.optionDict[a[0]] = a[1]
        optionList = [str(key) for key in self.optionDict.keys()]

        self.ui.optCB.addItems(optionList)
        self.ui.optCB.setCurrentText("input_ch")
        self.ui.optLE.setText(str(self.optionDict[self.ui.optCB.currentText()]))
        self.ui.optViewer.setText(str(self.optionDict[self.ui.optCB.currentText()]))

        self.ui.searchLE.setCompleter(QCompleter(optionList))
    
    def updateOptionCB(self):
        optionText = self.ui.searchLE.text() 
        if optionText in self.optionDict:
            self.ui.optCB.setCurrentText(optionText)

    def updateOptionLE(self):
        self.ui.optLE.setText(str(self.optionDict[self.ui.optCB.currentText()]))
        self.ui.optViewer.setText(str(self.optionDict[self.ui.optCB.currentText()]))

    def updateOption(self):
        key = self.ui.optCB.currentText()
        value = self.ui.optLE.text()
        prevValue = self.optionDict[key]

        def is_float(element):
            try:
                float(element)
                return True
            except ValueError:
                return False

        if is_float(prevValue) and not isinstance(prevValue, bool):
            if not is_float(value):
                self.popMessage("Option Input", "Wrong input! Input must be ({})".format(type(prevValue)))
                self.optLE.setText(str(prevValue))
            else:
                if isinstance(prevValue, int):
                    self.optionDict[key] = int(value)
                elif isinstance(prevValue, float):
                    self.optionDict[key] = float(value)
        else:
            if value.isnumeric():
                self.popMessage("Option Input", "Wrong input")
                self.optLE.setText(str(prevValue))
            else: 
                if isinstance(prevValue, bool):
                    if value in ["True", "False"]:
                        self.optionDict[key] = True if value == "True" else False
                    else:
                        self.popMessage("Option Input", "Wrong input! Input must be ({})".format(type(prevValue)))
                        self.optLE.setText(str(prevValue))
                else:
                    self.optionDict[key] = value

        if self.ui.optViewer.text() != self.optionDict[key]:
            self.ui.optViewer.setText(str(value))
                
        '''Update opt'''
        self.opt.__dict__.update(self.optionDict)

    def updateIndex(self):
        key = self.ui.indexCB.currentText()
        value = self.label_classes[key]
        self.ui.indexLE.setText(str(value))
    
    def updateLabel(self):
        if self.labelImage is None:
            if self.img is None:
                self.popMessage("Update Label", "No image to refer to!")
            else:
                self.labelImage = np.ones((self.img.shape[0],self.img.shape[1])).astype(np.uint8)
        labelImage = np.asarray(self.labelImage)
        mask = cv2.resize(self.mask,(labelImage.shape[1],labelImage.shape[0]),interpolation=cv2.INTER_NEAREST)
        print(np.count_nonzero(mask))
        labelImage[mask!=0] = int(self.ui.indexLE.text())
        self.labelImage = Image.fromarray(labelImage)
        self.ui.showLabelCB.setChecked(True)

    def updateDepth(self):
        if self.disp is None:
            self.popMessage("Update Depth", "No depth map!")
            return

        mask = cv2.resize(self.mask,(self.disp.shape[1],self.disp.shape[0]),interpolation=cv2.INTER_NEAREST) 
        self.disp[mask!=0] = self.ui.valueSlider.value() / 100.0 * 10
        self.toggleColormap()

    def krPath_imread(self, filePath) :
        stream = open(filePath.encode("utf-8") , "rb")
        bytes = bytearray(stream.read())
        numpyArray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)

    def loadImage(self, imageType):
        if self.fileName is None:
            fileName = QFileDialog.getOpenFileName(self, 'Open file', self.dirName + '../InferenceData/')[0]
        else:
            dirName, baseName = os.path.split(self.fileName)
            fileName = QFileDialog.getOpenFileName(self, 'Open file', dirName)[0]
        # fileName = QFileDialog.getOpenFileName(self, 'Open file', "R:/DCM_cropped/images_withGT")[0]
        if imageType == "image":
            if '.png' not in fileName and '.jpg' not in fileName: return
            self.ui.lineEdit.setText(fileName)

        elif imageType == "stroke":
            if '.png' not in fileName: return
            
        self.ui.statusBar.showMessage("Status: {} Loaded".format(imageType.capitalize()))
        
        
        if imageType == "image": 
            self.img = self.krPath_imread(fileName)
        img = QtGui.QImage(fileName)

        if img.width() > img.height():
            img2 = img.scaledToWidth(self.displaySize[0])
        else:
            img2 = img.scaledToHeight(self.displaySize[1])

        # Display image
        if imageType == "image":
            self.image = QtGui.QPixmap.fromImage(img2)
            self.ui.imageDisplay.setPixmap(self.image)

            self.fileName = fileName # Save filename for later inference
            
            # Create guide image
            self.guide = QtGui.QPixmap(img2.width(), img2.height())
            self.guide.fill(QtGui.QColor(0,0,0,0))
            # self.prev_guide = self.pixmap2numpy(self.guide)
            self.prev_guides = deque(maxlen=self.max_history)

            # Autoload label if exists
            head, tail = os.path.split(self.fileName)
            self.segFilename = glob.glob(os.path.splitext(os.path.join(head,'seg_' + tail))[0]+".*")
            self.segFilename = []
            print(self.segFilename, len(self.segFilename))
            if len(self.segFilename) == 0:
                self.segFilename = ''
                self.labelImage = Image.fromarray(
                    np.ones((self.img.shape[0],self.img.shape[1])).astype(np.uint8)*255)
            else:
                self.segFilename = self.segFilename[0]
                self.labelImage = Image.open(self.segFilename)
            
        # Update loaded stroke/guide to the image
        elif imageType == "stroke":
            self.guide = QtGui.QPixmap.fromImage(img2)
            results = self.join_pixmap(self.image, self.guide)
            self.ui.imageDisplay.setPixmap(results)
        
        # Re-initialize graphics
        self.ui.depthDisplay.clear()
        self.ui.progressBar.setValue(0)

        # Set infered data to None
        self.midas_depth = None

    def loadLabel(self):
        if self.fileName is None:
            filename = QFileDialog.getOpenFileName(self, 'Open file', self.dirName + '../InferenceData/')[0]
        else:
            dirName, baseName = os.path.split(self.fileName)
            filename = QFileDialog.getOpenFileName(self, 'Open file', dirName)[0]
        self.ui.labelLE.setText(str(filename))
        self.segFilename = filename

    def processImage(self):
        if self.ui.imageDisplay.pixmap() is None:
            self.popMessage("Process Image", "No image to process! Please load an image")
            return

        # check whehter to use stroke or not
        if self.ui.noStrokeCB.isChecked():
            guide = np.zeros((self.image.height(), self.image.width(), 4),dtype=np.uint8) # H x W x 4
        else:
            guide = self.pixmap2numpy(self.guide)
        
        if self.mode == 'refine':
            guide -= self.prev_guide
            self.prev_guide = self.pixmap2numpy(self.guide)

        guide_alpha = guide[:,:,-1]
        guide_input = guide[:,:,:3]

        self.ckpt = self.ui.ckptCB.currentText()
        self.params['option'] = self.opt

        self.params['label_type'] = "no_label"

        self.processTask = TaskThread(self.fileName, 
                                    self.segFilename,
                                    guide_input, 
                                    guide_alpha, 
                                    self.mode, 
                                    self.ckpt, 
                                    False, 
                                    self.params)

        self.processTask.load_label_model(self.labelModel)
        self.processTask.label = self.labelImage
        if not self.model is None:
            self.processTask.load_model(self.model)
        if not self.midas_depth is None:
            self.processTask.midas_depth = self.midas_depth
        self.processTask.timeSignal.connect(self.onProgress)
        
        if self.userstudyTask is not None:
            self.userstudyTask.pause()
        self.processTask.start()

    def initProgress(self, position):
        if position == 100:
            self.processTask.stop()
            return   

    def onProgress(self, position):
        if position != -1:
            self.ui.progressBar.setValue(position)
        else:
            disp = self.disp/10*255
            self.processTask.saveImage("D:/"+self.ui.usernameLE.text()+".png",disp)
        self.ui.statusBar.showMessage("Status: Estimating Depth")
        if position == 100:
            self.disp = self.processTask.disp
            if self.disp is None:
                QtWidgets.QMessageBox.information(self, "ERROR", "Load the image")
                return
            
            self.processTask.stop()
            # self.processTask.resumeTimer()
            # self.processTask.start()
            self.model = self.processTask.model
            self.labelImage = self.processTask.label

            if self.midas_depth is None:
                self.midas_depth = self.processTask.midas_depth
            else:
                self.midas_depth = torch.Tensor(self.disp).unsqueeze(0).unsqueeze(0)

            ''' Resize depth image to orginal size '''
            print("Disp shape to image shape ", self.img.shape)
            self.disp = cv2.resize(self.disp, (self.img.shape[1], self.img.shape[0]),cv2.INTER_NEAREST)
            
            cv2.imwrite('result.png',self.disp)
           
            ''' Resize depth image to viewing display size '''
            viewImage = self.ui.imageDisplay.pixmap()
            viewDisp = cv2.resize(self.disp, (viewImage.width(),viewImage.height()))
            # viewDisp = np.transpose(self.disp, (1,0))
            # viewDisp = cv2.resize(viewDisp, (viewImage.width(),viewImage.height())).astype(np.uint8)
            viewDisp = ((viewDisp/10.0)*255).astype(np.uint8)
            # viewDisp = ((viewDisp-viewDisp.min())/(viewDisp.max()-viewDisp.min()) * 255).astype(np.uint8)

            if self.ui.showColormapCB.isChecked():
                inferno = plt.get_cmap('inferno')
                viewDisp = inferno(viewDisp/255.0)*255
                disp_ = qimage2ndarray.array2qimage(viewDisp)
            else:
                disp_ = QtGui.QImage(viewDisp.data, \
                    viewImage.width(), 
                    viewImage.height(),
                    viewDisp.strides[0],
                QtGui.QImage.Format_Indexed8)

            ''' Display depth '''
            self.ui.depthDisplay.setPixmap(QtGui.QPixmap.fromImage(disp_))
            self.ui.statusBar.showMessage("Status: Depth Displayed!")

            if self.userstudyTask is not None:
                self.userstudyTask.resume()
                self.userstudyTask.start()
            
    def saveImage(self, saveType):
        if saveType == "depth" and self.ui.depthDisplay.pixmap() is None:
            self.popMessage("Save Image", "No disparity image to save!")
            return
        elif saveType == "stroke" and self.guide == None:
            self.popMessage("Save Image", "No stroke to save!")
            return
        elif saveType == "label" and self.guide == None:
            self.popMessage("Save Image", "No label to save!")
            return
        
        saveName = QFileDialog.getSaveFileName(self, 'Save file', self.dirName)[0]
        
        if ".png" in saveName:
            if saveType == 'depth':
                disp = self.disp/10*255
                cv2.imwrite(saveName, disp.astype("uint8"))
            elif saveType == 'stroke':
                img = self.guide
                img.save(saveName,'PNG')
            elif saveType == 'label':
                img = self.labelImage
                img.save(saveName,'PNG')
        elif ".npy" in saveName and saveType == 'depth':
            np.save(saveName, self.disp)
        else:
            self.popMessage("saveImage","Wrong Format!")

        self.ui.statusBar.showMessage("Status: {} Image Saved".format(saveType.capitalize()))

    def plotImage(self):
        if self.figureWidget is None:
            self.figureWidget = FigureWidget()
        
        if not self.fileName is None and not self.disp is None:
            img = np.array(Image.open(self.fileName))
            h,w,c = img.shape
            ratio = 384 / max(w,h)
            self.figureWidget.plot_pyqtgraph(img, self.disp, ratio)

        self.figureWidget.show()

    def toggleLabelImage(self):
        if self.ui.showLabelCB.isChecked():
            if self.labelImage is None:
                self.ui.showLabelCB.toggle()
                self.popMessage("Error", "No label to display!")
                return
            labelImage = self.labelImage.toqimage()
            labelImage = labelImage.scaledToWidth(self.image.width())
            self.ui.imageDisplay.setPixmap(QtGui.QPixmap.fromImage(labelImage))
        else:
            results = self.join_pixmap(self.image, self.guide)
            self.ui.imageDisplay.setPixmap(results)

    def toggleGuideImage(self):
        if not self.ui.showStrokeCB.isChecked():
            if self.guide is None:
                self.ui.showStrokeCB.toggle()
                self.popMessage("Error", "No stroke to display")
                return
            self.ui.imageDisplay.setPixmap(self.image)
        else:
            results = self.join_pixmap(self.image, self.guide)
            self.ui.imageDisplay.setPixmap(results)

    def toggleDepthImage(self):
        if self.ui.showDepthCB.isChecked():
            if self.disp is None:
                self.popMessage("Error", "No label to display!")
                return
            dispImage = self.ui.depthDisplay.pixmap()
            results = self.join_pixmap(dispImage, self.guide)
            self.ui.imageDisplay.setPixmap(results)
        else:
            results = self.join_pixmap(self.image, self.guide)
            self.ui.imageDisplay.setPixmap(results)
    
    def toggleColormap(self):
        if self.disp is None:
            return
        viewImage = self.ui.imageDisplay.pixmap()
        viewDisp = cv2.resize(self.disp, (viewImage.width(),viewImage.height()))
        viewDisp = ((viewDisp-viewDisp.min())/(viewDisp.max()-viewDisp.min()) * 255).astype(np.uint8)

        if self.ui.showColormapCB.isChecked():
            inferno = plt.get_cmap('inferno')
            viewDisp = inferno(viewDisp/255.0)*255
            plt.imsave("colormap.png",viewDisp/255)
            disp_ = qimage2ndarray.array2qimage(viewDisp)
        else:
            disp_ = QtGui.QImage(viewDisp.data, \
                viewImage.width(), 
                viewImage.height(),
                viewDisp.strides[0],
            QtGui.QImage.Format_Indexed8)

        ''' Display depth '''
        self.ui.depthDisplay.setPixmap(QtGui.QPixmap.fromImage(disp_))

    def toggleOverlay(self):
        if self.image is None:
            return
        if self.ui.showOverlayCB.isChecked():
            viewImage = self.ui.imageDisplay.pixmap()
            dispImage = self.ui.depthDisplay.pixmap()
            result = self.join_pixmap(dispImage,viewImage,mode=QtGui.QPainter.CompositionMode_Overlay)
            self.ui.depthDisplay.setPixmap(result)

        else:
            self.toggleColormap()

    def pixmap2numpy(self, pixmap):  
        qimg = pixmap.toImage()
        image = qimage2ndarray.rgb_view(qimg)
        alpha = np.expand_dims(qimage2ndarray.alpha_view(qimg),2)
        image = np.concatenate((image,alpha),axis=2)
        return image
    

    def changeValue(self):
        size = self.ui.valueSlider.value() / 100.0
        self.ui.valueLabel.setText(str(size))

        color = int(size*255)
        self.pen_color = QtGui.QColor(color,color,color, 255)

        self.updateValueBox(color)

    def changeSize(self):
        size = self.ui.sizeSlider.value()
        self.ui.sizeLabel.setText(str(size))

        self.pen_size = size
        self.brushPixmap = QtGui.QPixmap(self.pen_size,self.pen_size)
        self.brushPixmap.fill(self.pen_color)
        self.ui.valueBox.setPixmap(self.brushPixmap)

    def updateSlider(self, value):
        self.ui.valueSlider.setValue(value)

    def updateValueBox(self, color):
        self.brushPixmap.fill(QtGui.QColor(color,color,color))
        self.ui.valueBox.setPixmap(self.brushPixmap)

    def mousePressEvent(self, e):
        # Check if any image is loaded
        if self.ui.imageDisplay.pixmap() is None:
            return
        # Check if the click happens within display boxes
        isImageDisplay = True
        if self.inBoundary(e.pos(), self.ui.imageBox): 
            # If current mouse pointer is out of image box
            localPos = self.ui.imageBox.mapFromParent(e.pos())
            localPos = self.ui.imageDisplay.mapFromParent(localPos)
            # if it is a stroke mode
            if not self.pickerActive:
                self.prev_guides.append(self.guide)
        elif self.inBoundary(e.pos(), self.ui.depthBox) and self.pickerActive \
            and self.ui.depthDisplay.pixmap() is not None: 
            # If current mouse pointer is out of depth box and color picker mode isn't active
            localPos = self.ui.depthBox.mapFromParent(e.pos())
            localPos = self.ui.depthDisplay.mapFromParent(localPos)
            isImageDisplay = False
        else:
            self.last_x = None
            self.last_y = None
            return

        img_size = self.ui.imageDisplay.pixmap().size()
        x = int(localPos.x() - (self.displaySize[0]-img_size.width())/2)
        y = int(localPos.y() - (self.displaySize[1]-img_size.height())/2)

        #mouse is pressed while shift key is pressed
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            print("shift pressed")
            stroke = QtGui.QPixmap(self.guide.width(), self.guide.height())
            stroke.fill(QtGui.QColor(0,0,0,0))
            painter = QtGui.QPainter(stroke)
            p = painter.pen()
            p.setWidth(self.pen_size)
            p.setColor(self.pen_color)
            painter.setPen(p)
            painter.drawLine(self.last_x, self.last_y, x, y)
            painter.end()
            self.update()

            # if eraser is checked, change the mode for clearing out the original guide
            # if not, composite current stroke to guide
            if self.ui.eraseButton.isChecked():
                mode = QtGui.QPainter.CompositionMode_DestinationOut
            else:
                mode = QtGui.QPainter.CompositionMode_SourceOver
            self.guide = self.join_pixmap(self.guide, stroke, mode=mode)
            
            # Composite guide with loaded rgb image for visualization
            results = self.join_pixmap(self.image, self.guide)
            self.ui.imageDisplay.setPixmap(results)

            #log
            self.log('mousePress','line')
        
        # Update the origin for next time.
        self.last_x = x
        self.last_y = y
       
        if self.last_x is None:
            self.last_x = x
            self.last_y = y

        if self.pickerActive: #color picker is currently active
            if isImageDisplay:
                value = self.guide.toImage().pixelColor(x,y).red()
            else:
                value = self.ui.depthDisplay.pixmap().toImage().pixelColor(x,y).red()
            self.updateSlider(int(value/255*100))
            self.updateValueBox(value)
            return


    def inBoundary(self, pos, constraint):
        if type(constraint) == list:
            # Constraint is a list of boundary
            x_min, x_max, y_min, y_max = constraint
        else:
            # constraint is an object
            x_min, y_min = constraint.x(), constraint.y()
            x_max, y_max = x_min + constraint.width(), y_min + constraint.height()

        if pos.x() > x_min and pos.x() < x_max \
            and pos.y() > y_min and pos.y() < y_max:
            return True
        else:
            return False
       
    def mouseMoveEvent(self, e):
        if e.buttons() == QtCore.Qt.NoButton:
            return
        
        # Check if any image is loaded
        if self.ui.imageDisplay.pixmap() is None:
            return
        # Check if the click happens within display boxes
        isImageDisplay = True
        if self.inBoundary(e.pos(), self.ui.imageBox): 
            # If current mouse pointer is out of image box
            localPos = self.ui.imageBox.mapFromParent(e.pos())
            localPos = self.ui.imageDisplay.mapFromParent(localPos)
        elif self.inBoundary(e.pos(), self.ui.depthBox) and self.pickerActive \
            and self.ui.depthDisplay.pixmap() is not None: 
            # If current mouse pointer is out of depth box and color picker mode isn't active
            localPos = self.ui.depthBox.mapFromParent(e.pos())
            localPos = self.ui.depthDisplay.mapFromParent(localPos)
            isImageDisplay = False
        else:
            return

        img_size = self.ui.imageDisplay.pixmap().size()
        x = int(localPos.x() - (self.displaySize[0]-img_size.width())/2)
        y = int(localPos.y() - (self.displaySize[1]-img_size.height())/2)

        if self.last_x is None:
            self.last_x = x
            self.last_y = y
            return

        if self.pickerActive: #color picker is currently active
            if isImageDisplay:
                value = self.guide.toImage().pixelColor(x,y).red()
            else:
                value = self.ui.depthDisplay.pixmap().toImage().pixelColor(x,y).red()
            self.updateSlider(int(value/255*100))
            self.updateValueBox(value)
            return
        elif self.ui.fillButton.isChecked():
            return
            
        # Create current stroke layer
        stroke = QtGui.QPixmap(self.guide.width(), self.guide.height())
        stroke.fill(QtGui.QColor(0,0,0,0))

        painter = QtGui.QPainter(stroke)
        p = painter.pen()
        p.setWidth(self.pen_size)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, x, y)
        painter.end()
        self.update()

        # if eraser is checked, change the mode for clearing out the original guide
        # if not, composite current stroke to guide
        if self.ui.eraseButton.isChecked():
            mode = QtGui.QPainter.CompositionMode_DestinationOut
        else:
            mode = QtGui.QPainter.CompositionMode_SourceOver
        self.guide = self.join_pixmap(self.guide, stroke, mode=mode)
        
        # Composite guide with loaded rgb image for visualization
        results = self.join_pixmap(self.image, self.guide)
        self.ui.imageDisplay.setPixmap(results)
        
        # Update the origin for next time.
        self.last_x = x
        self.last_y = y
    
    def changeThreshold(self):
        if self.ui.selectButton.isChecked():
            self.segmentArea(self.last_select_x,self.last_select_y,self.ui.thresholdSB.value())

    def segmentArea(self,x,y, threshold):
        disp = self.pixmap2numpy(self.ui.depthDisplay.pixmap())
        mask = np.zeros((disp.shape[0],disp.shape[1],4))
        mask = flood_fill(disp,mask,int(x),int(y),QtGui.QColor('#FF0000'),threshold=threshold)
        results = self.join_pixmap(self.image, QtGui.QPixmap.fromImage(Image.fromarray(mask.astype(np.uint8)).toqimage()))
        self.ui.imageDisplay.setPixmap(results)
        self.mask = mask[:,:,0] # R channel == 1
    
    def mouseReleaseEvent(self, e):
        if self.ui.strokeButton.isChecked():
            self.log('mouseRelease','stroke')
        elif self.ui.eraseButton.isChecked():
            self.log('mouseRelease','erase')
        if self.pickerActive and not (self.last_x is None and self.last_y is None):
            self.pickerActive = False
            # QApplication.restoreOverrideCursor()
            QApplication.setOverrideCursor(Qt.ArrowCursor)
            self.ui.strokeButton.setChecked(True)
            self.activateStroke()
            #log
            self.log('mouseRelease','picker')
        if self.ui.fillButton.isChecked() and not (self.last_x is None and self.last_y is None):
            guide = self.pixmap2numpy(self.guide)
            if self.userstudyTask is not None:
                self.userstudyTask.pause()
            guide = flood_fill(guide,guide.copy(),int(self.last_x),int(self.last_y),self.pen_color)
            self.guide = QtGui.QPixmap.fromImage(Image.fromarray(guide).toqimage())
            results = self.join_pixmap(self.image, self.guide)
            self.ui.imageDisplay.setPixmap(results)
            #log
            if self.userstudyTask is not None:
                self.userstudyTask.resume()
                self.userstudyTask.start()
            self.log('mouseRelease','fill')
            

        # self.last_x = None
        # self.last_y = None
        
        if self.inBoundary(e.pos(), self.ui.imageBox) and not self.pickerActive:
            self.processImage()


    # def keyPressEvent(self, e):
    #     if e.key() == Qt.Key_Escape:
    #         self.close()
    #     elif e.key() == Qt.Key_Z and e.modifiers() == Qt:
    #         self.loadImage()
    #     elif e.key() == Qt.Key_N:
    #         self.showNormal()

        
    def join_pixmap(self, p1, p2, mode=QtGui.QPainter.CompositionMode_SourceOver):
        result =  QtGui.QPixmap(p1)
        result.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(result)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.drawPixmap(result.rect(), p1, p1.rect())
        painter.setCompositionMode(mode)
        painter.drawPixmap(result.rect(), p2, p2.rect())
        painter.end()
        return result

    def undoStroke(self):
        if len(self.prev_guides) == 0:
            QtWidgets.QMessageBox.information(self, "Undo", "No more history to undo!")
            return
        self.prev_guide = self.prev_guides.pop()
        results = self.join_pixmap(self.image, self.prev_guide)
        self.ui.imageDisplay.setPixmap(results)
        self.guide = self.prev_guide

        self.processImage()

    def eraseAll(self):
        if self.ui.imageDisplay.pixmap() is None:
            self.popMessage("Erase All", "Nothing to clear!")
            return

        self.prev_guides.append(self.guide)
        self.guide = QtGui.QPixmap(self.guide.width(), self.guide.height())
        self.guide.fill(QtGui.QColor(0,0,0,0))

        # Update image
        results = self.join_pixmap(self.image, self.guide)
        self.ui.imageDisplay.setPixmap(results)
        
        self.processImage()
        print(self.prev_guide, self.guide)
        print("Erase All Strokes")

    def activateStroke(self):
        self.ui.eraseButton.setChecked(False)

        size = self.ui.valueSlider.value() / 100.0
        color = size*255
        self.pen_color = QtGui.QColor(color,color,color, 255)

    def activateEraser(self):
        self.ui.strokeButton.setChecked(False)
    
    def activateFill(self):
        self.ui.strokeButton.setChecked(False)
        self.ui.eraseButton.setChecked(False)

        size = self.ui.valueSlider.value() / 100.0
        color = size*255
        self.pen_color = QtGui.QColor(color,color,color, 255)
    
    def activateSelect(self):
        if self.ui.selectButton.isChecked():
            self.ui.imageDisplay.setPixmap(self.image)
        else: #deactivate
            self.ui.imageDisplay.setPixmap(self.join_pixmap(self.image, self.guide))
            self.ui.showLabelCB.setChecked(False)


    def activatePicker(self, event):
        if self.ui.imageDisplay.pixmap() is None:
            self.popMessage("Active Picker", "No image to pick value! Please load an image")
            return
        
        self.ui.strokeButton.setChecked(False)
        self.ui.eraseButton.setChecked(False)
        self.pickerActive = True

        QApplication.setOverrideCursor(Qt.PointingHandCursor)
        self.ui.statusBar.showMessage("Status: Pick a Disparity Value")
        
    def popMessage(self, title, message):
        self.msg.setWindowTitle(title)
        self.msg.setText(message)
        self.msg.exec_()

    def updateMode(self):
        self.ui.ckptCB.clear()
        self.mode = "segDepth"
        ckptList = sorted([os.path.basename(x) for x in glob.glob("./ckpt/"+self.mode+"/*.pth")])
        self.ui.ckptCB.addItems(ckptList)

        self.model = None

    def refreshModel(self):
        self.model = None

    '''userstudy functions'''
    def toggleLogging(self):
        if self.ui.userLogCB.isChecked():
            self.ui.usernameLE.setEnabled(True)
            self.ui.userStartButton.setEnabled(True)
            self.ui.userDoneButton.setEnabled(True)
        else:
            self.ui.usernameLE.setEnabled(False)
            self.ui.userStartButton.setEnabled(False)
            self.ui.userDoneButton.setEnabled(False)
    def startLogging(self):
        print("(Start Logging)")
        path = "D:/Dropbox/01_Work/VML/03_Research/03_Seg2Depth/99_experiment/02_userstudy/02_usability/01_result/M1_ours/"
        name = self.ui.usernameLE.text()
        self.logDirname = os.path.join(path,name,os.path.basename(self.fileName).split(".")[0])
        print(self.logDirname)
        if not os.path.exists(self.logDirname):
            print("make dirs!")
            os.makedirs(self.logDirname) 
        self.userstudyTask = UserStudyThread(os.path.join(self.logDirname, "log.txt"))
        self.userstudyTask.timeSignal.connect(self.logOnProgress)
        self.userstudyTask.start()
    
    def stopLogging(self):
        self.userstudyTask.stop()
        disp = self.disp/10*255
        dtime = datetime.now().strftime('%H%M%S')
        savename = os.path.join(self.logDirname,dtime +'_final.png')
        cv2.imwrite('D:/Dropbox/test.png', disp.astype("uint8"))
        self.popMessage("Logging", "Loggig Stopped!")
        print("(Stop Logging)   Userstudy Stopped!")
        self.userstudyTask.quit()
        self.userstudyTask = None
        
    
    def logOnProgress(self, position):
        if position == 50:
            if self.disp is not None:
                disp = self.disp/10*255
                dtime = datetime.now().strftime('%H%M%S')
                savename = os.path.join(self.logDirname,dtime +'.png')
                cv2.imwrite(savename, disp.astype("uint8"))

    def log(self,eventType,action='stroke'):
        if self.userstudyTask is None:
            return
        print("!!!!!!!!!!!!",eventType, action)
        self.userstudyTask.log('{}::{}'.format(eventType, action))
        


import data
from models.depth_model import DepthModel
import models.networks.generator as generator

from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose
import torch 
import csv
import util.util as util

class TaskThread(QtCore.QThread):
    timeSignal = QtCore.pyqtSignal(int)
    status = 'run'
    f = None
    startTime = None
    def __init__(self, filename= None, segFilename = None, guide = None, alpha = None, mode = "depth", ckptName ='latest_net_G.pth', isOptimize = False, params = {}):
        super(TaskThread, self).__init__()

        self.filename = filename
        self.segFilename = segFilename
        self.disp = None
        self.guide = guide #numpy array
        self.alpha = alpha
        self.mode = mode
        self.ckptName = ckptName
        self.label = None

        self.dispScale = 10
        self.isOptimize = isOptimize
        self.params = params # ui other parameters
        '''Things that are used over an over again if ckpt/image is not changed'''
        self.model = None
        self.midas_depth = None

    def load_model(self, model):
        self.model = model

    def load_label_model(self, labelModel = None):
        if labelModel is None:
            device = 'cpu' if self.params['option'].gpu_ids == [] else 'cuda'
            print(device)
            # self.labelModel = util.SenFormerInference(device)
            self.timeSignal.emit(100)
            print("(Load Label Model) Loading Complete")
        else:
            self.labelModel = labelModel

    def run(self):
        if self.status == "launch":
            self.load_label_model(None)
            return

        if self.filename is None:
            self.timeSignal.emit(100)
            return
        self.timeSignal.emit(1)

        # opt = TestOptions().parse()
        opt = self.params['option']

        head, tail = os.path.split(self.filename)
        
        '''Deprecated since 220829'''
        # self.segFilename = os.path.join(head,'seg_' + tail).replace(".jpg",".png")
        # if not os.path.isfile(self.segFilename):
        #     self.segFilename = self.segFilename.replace(".png", ".jpg")
        # self.instFilename = os.path.join(head, 'inst_' + tail)
        '''-----------------------'''

        ckpt = './ckpt/'+ self.mode + "/" + self.ckptName

        # weights = torch.load(ckpt)
        # model.load_state_dict(weights)
        # # import models.networks.encoder as encoders
        # # model.seg_encoder = encoders.HRNetEncoder()
        # # model.seg_decoder.load_state_dict(
        # #         torch.load(weight, map_location=lambda storage, loc: storage), strict=False)
        # model.eval()
        
        img = Image.open(self.filename)
        print("original input size: ", np.array(img).shape)

        if opt.decoder == "spade" and opt.encoder == "MobileNetV2":
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
            # gray_blur = cv2.GaussianBlur(gray,(3,3), sigmaX=0, sigmaY=0)
            edge = cv2.Canny((gray).astype(np.uint8), 100, 200)
            img = Image.fromarray(np.stack((edge,)*3, axis=-1))

        '''---before 1229---
        target_size = 256
        if opt.experiment in ["MiDaS","DPT-Large"] or opt.encoder == "MiDaS":
            target_size = 384
        ratio = target_size / min(img.size[1],img.size[0])
        input_size = (int(img.size[1] * ratio // 32 * 32),int(img.size[0] * ratio // 32 * 32))
        # input_size = (target_size,target_size)
        '''
        target_size = opt.infer_size
        if opt.fit_max:
            ratio = target_size / max(img.size[1],img.size[0])
        else:
            ratio = target_size / min(img.size[1],img.size[0])
        input_size = (int(img.size[1] * ratio // 32 * 32),int(img.size[0] * ratio // 32 * 32))
        
        print("transform to size: ", input_size)
        transform = transforms.Compose([
            transforms.Resize(input_size, interpolation = Image.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225])
        ])
        '''if input image has to be normalized'''
        if opt.experiment in ["MiDaS"] or opt.encoder == 'MiDaS':
            if "localloss" not in self.ckptName:
                img = transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                            std=[0.229, 0.224, 0.225]) \
                                            (transform(img)[0:3,:,:]).unsqueeze(0) #* (1.0 / 255.0)
                print("(Interface) Image is normalized")
            else:
                img = transform(img)[0:3,:,:].unsqueeze(0)
        elif opt.experiment in ["DPT-Large"]:
            print("normalize by 0.5")
            img = transforms.Normalize(mean=[0.5, 0.5, 0.5],\
                                            std=[0.5, 0.5, 0.5]) \
                                            (transform(img)[0:3,:,:]).unsqueeze(0)
        else:
            img = transform(img)[0:3,:,:].unsqueeze(0)
        print("Image size: ",img.shape)

        guide = Image.fromarray(self.guide)
        self.alpha = np.tile(np.expand_dims(self.alpha,2), (1,1,3))
        mask = Image.fromarray(self.alpha)
        transform = transforms.Compose([
            transforms.Resize(input_size, interpolation = Image.NEAREST),
            transforms.ToTensor(),
        ])
        guide = transform(guide)[0,:,:] * self.dispScale
        mask = transform(mask)[0,:,:]

        if opt.guide_empty == -1.0:
            guide[mask==0] = -1.0
        elif opt.guide_empty == 0.0:
            guide[guide==0.0] = 0.1; guide[mask==0] = 0.0
        guide = guide.reshape([1,1,guide.shape[-2],guide.shape[-1]])

        if opt.no_guide:
            input = img

        elif opt.mask_rgb_with_guide:
                masked_rgb = img * torch.cat((((guide==0)*1.0,)*3),dim=1)
                input = torch.cat([masked_rgb, guide], dim=1)
        else:
            input = torch.cat([img, guide], dim = 1)
        
        if self.mode == 'refine':
            if self.midas_depth is None:
                self.midas_depth = self.infer_MiDaS(self.filename)
                self.midas_depth = transform(Image.fromarray(self.midas_depth))[0:3,:,:].unsqueeze(0)
            input = torch.cat([input,self.midas_depth], dim=1)
        input_semantics = None
        print("label_type: ", self.params['label_type'])
        if self.mode == 'segMap' or (self.mode == 'segDepth' and not opt.infer_segmap):
            transform = transforms.Compose([
                transforms.Resize(input_size, interpolation = Image.NEAREST),
                transforms.ToTensor(),
            ])
            if  self.params['label_type'] == "no_label" or \
                (self.params['label_type'] == "load_label" and not os.path.isfile(self.segFilename)):
                if "ADE" in self.ckptName or 'ade' in self.ckptName:
                    label = Image.fromarray(np.zeros((img.size()[1],img.size()[0]),dtype=np.uint8)*255)
                else: label = Image.fromarray(np.ones((img.size()[1],img.size()[0]),dtype=np.uint8)*255)
            else:
                if self.params['label_type'] == "load_label" and os.path.isfile(self.segFilename): 
                    label = Image.open(self.segFilename)
                    self.label = label
                elif self.params['label_type'] == "infer_label":
                    print("Generating label from a file: ", self.filename)
                    label = self.labelModel.infer(self.filename)
                    label = util.convert_label(label)
                    label = np.stack((label,label,label), axis=-1).astype(np.uint8)
                    #color label
                    for idx in np.unique(label):
                        cmap = util.labelcolormap(182)
                        mask = label[:,:,1]==idx
                        label[mask][0]= cmap[idx]
                        label[mask][1]=cmap[idx]
                        label[mask][2]=cmap[idx]

                    label = Image.fromarray(label)
                    self.label = label
                else:
                    label = self.label
            
            label = transform(label)[0].unsqueeze(0).unsqueeze(0) * 255.0
            
            # if self.params['label_type'] == "load_label":
            #     label[label>0] = label[label>0] -1

            label[label>opt.label_nc] = opt.label_nc

            '''check types of label'''
            # from collections import Counter
            # print(Counter(torch.flatten(label).numpy()))
            
            # instance = Image.open(self.instFilename)
            # instance = transform(instance)[0].unsqueeze(0).unsqueeze(0) * 255.0
            # instance = instance.long()

            label_map = label.long()
            bs, _, h, w = label_map.size()
            nc = opt.label_nc + 1 if opt.contain_dontcare_label \
                else opt.label_nc
            
            self.FloatTensor = torch.FloatTensor
            self.ByteTensor = torch.ByteTensor
            input_label = self.FloatTensor(1, nc, h, w).zero_()

            input_semantics = input_label.scatter_(1, label_map, 1.0)
            
            if not opt.no_instance:
                instance_edge_map = self.get_edges(label)
                if opt.debug:
                    print("debugging")
                    edge = np.load("edge.npy")
                    instance_edge_map = transform(Image.fromarray(edge))[0].unsqueeze(0).unsqueeze(0)
                # instance_edge_map = torch.zeros_like(label)

                input_semantics = torch.cat((input_semantics, instance_edge_map),dim=1)
            input = torch.cat([input, input_semantics], dim=1)

        self.timeSignal.emit(50)
        
        # inference
        # disp = model(input)
        # self.disp = disp.detach().squeeze().numpy()

        from infer import Inference
        inference = Inference(opt)
        if self.model is None:
            inference.load_model(ckpt, self.mode)
            self.model = inference.model
        else:
            inference.model = self.model
        print("(Interface)  Inference start with mode: {} ".format(self.mode))
        print("input shape ", input.shape)
        device = 'mps' if self.params['option'].gpu_ids == [] else 'cuda'
        print("!!! device:",device)
        inference.model.to(device); input= input.to(device)
        start = time.time()
        self.disp = inference.infer(input, self.isOptimize)
        print("(Interface)  Execution time: {}".format(time.time()-start))
        self.timeSignal.emit(100)

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()  
    
    def convert_label(self, filename):
        # Mapping table from coco to ade20k
        labelTable = csv.DictReader(open("../label.csv"))
        table = {}
        for row in labelTable:
            name, cocoIdx, adeIdx = row['Name'], row['Coco'], row['ADE20K']
            if cocoIdx != '':
                table[int(cocoIdx)] = int(adeIdx) if adeIdx != '' else -1

        transform = transforms.Compose([
                transforms.Resize((64,64), interpolation = Image.NEAREST),
                transforms.ToTensor(),
            ])
        label = Image.open(filename)
        label = transform(label)[0].unsqueeze(0).unsqueeze(0) * 255.0
        label[label>0] = label[label>0] - 1
        label[label == 255] = 182

        # Convert
        #tobecontinue

        label_map = label.long()
        bs, _, h, w = label_map.size()
        nc = opt.label_nc + 1 if opt.contain_dontcare_label \
            else opt.label_nc
        
        self.FloatTensor = torch.FloatTensor
        self.ByteTensor = torch.ByteTensor
        input_label = self.FloatTensor(1, nc, h, w).zero_()

        input_semantics = input_label.scatter_(1, label_map, 1.0)

    def infer_MiDaS(self, filename, model_type = "DPT_Large"):
        #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        print(filename, " is a Filename")
        img = np.array(Image.open(filename))[:,:,:3]
        # img = cv2.imread(filename)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        return output
        
    def __del__(self):
        self.wait()
        print('process thread destruct ...')

    def stop(self):
        print("Task stopped")
        self.wait()
        pass


from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtWebEngineWidgets
import plotly.graph_objs as go
import plotly
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class FigureWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        print("Initialize Figure Widget")

        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setWindowTitle("Point Cloud Display")

        
        # self.browser = QtWebEngineWidgets.QWebEngineView(self)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.widget = gl.GLViewWidget()
        self.layout.addWidget(self.widget)
        self.widget.setBackgroundColor('w')
        self.widget.show()
        # layout.addWidget(self.canvas)
        # self.layout.addWidget(self.browser)
        self.resize(500,500)

    def process_coordinates(self,img,disp,ratio):
        img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        disp = cv2.resize(disp, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        h,w,c = img.shape

        cols = np.array([[h-i]*w for i in range(h)]).flatten()
        rows = np.array([i for i in range(w)]*h)
        depth = disp.reshape(h*w,1)*-1
        colors = img.reshape((h*w,c))/255.0
        colors = np.concatenate(colors, np.ones((h*w,1)), axis=1) #alpha

        return cols, rows, depth, colors


    def plot(self, img, disp, ratio):
        self.ax.clear()

        img = np.array(img)

        img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        disp = cv2.resize(disp, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        # plt.imshow(disp); plt.show()
        h,w,c = img.shape

        cols = np.array([[h-i]*w for i in range(h)]).flatten()
        rows = np.array([i for i in range(w)]*h)
        # depth = np.array([1 for i in range(h*w)])
        depth = disp.reshape(h*w,1)*-1
        # depth = (depth*-1)+depth.max()
        colors = img.reshape((h*w,c))/255.0
        
        self.ax.set_box_aspect((1,1,h/w))
        self.ax._axis3don = False
        
        self.ax.scatter(xs = rows, ys = depth, zs = cols, c=colors, depthshade=0)
        self.canvas.draw()

    def plot_plotly(self,img,disp,ratio):
        print("plotly shown")
        ratio=0.1
        img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        disp = cv2.resize(disp, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        # plt.imshow(disp); plt.show()
        h,w,c = img.shape

        rows = np.array([[h-i]*w for i in range(h)]).flatten()
        cols = np.array([i for i in range(w)]*h)
        # depth = np.array([1 for i in range(h*w)])
        depth = disp.reshape(h*w,)
        # depth = (depth*-1)+depth.max()
        colors = img.reshape((h*w,c))/255.0
        print(cols.shape, rows.shape, disp.shape, depth.shape, colors.shape)

        data = go.Scatter3d(x=depth,y=cols,z=rows,mode='markers', marker=dict(size=3, color=colors))

        layout = go.Layout(
            autosize=False,
            width=500,
            height=500,
        )
        # fig = go.Figure(data=[data], layout=layout)
        fig = go.Figure()
        fig.add_trace(data)
        html = '<html><head><meta charset="utf-8" />'
        html += '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>'
        html += '<body>'
        html += plotly.offline.plot([data], include_plotlyjs=False, output_type='div')
        html += '</body></html>'
        # self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))
        self.browser.setHtml(html)
    def plot_pyqtgraph(self,img,disp,ratio):
        self.widget.clear()

        img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        disp = cv2.resize(disp, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        # plt.imshow(disp); plt.show()
        h,w,c = img.shape

        rows = np.array([[h-i]*w for i in range(h)]).flatten()
        cols = np.array([i for i in range(w)]*h)
        # depth = np.array([1 for i in range(h*w)])
        depth = disp.reshape(h*w,)*-1
        depthScale = min(h,w) / abs(min(depth))
        depth*= depthScale
        # depth = (depth*-1)+depth.max()
        colors = img.reshape((h*w,c))/255.0

        
        print(self.widget.cameraParams())
        self.widget.opts['distance'] = min(h,w) * 2
        # self.widget.orbit(-135,-30)
        self.widget.opts['elevation'] = 0
        self.widget.opts['azimuth'] = -90
        pos = np.array([[cols[i],depth[i],rows[i]] for i in range(len(depth))])
        # pos[:,2] = np.abs(pos[:,2])

        sp2 = gl.GLScatterPlotItem(pos=pos,size=3)
        sp2.setData(color=colors)
        sp2.setGLOptions('opaque')
        sp2.translate(-1*max(cols)/2,-1*min(depth)/2,-1*max(rows)/2)
        self.widget.addItem(sp2)

        
        print(self.widget.cameraParams())

 
class UserStudyThread(QtCore.QThread):
    timeSignal = QtCore.pyqtSignal(int)
    status = 'run'

    f = None
    startTime = None
    saveTime = 60

    def __init__(self, filename):
        super(UserStudyThread, self).__init__()
        write_mode = 'w'
        if os.path.exists(filename):
            write_mode = 'a'
        self.f = open(filename, write_mode)
        self.dtime = datetime.now().strftime('%H:%M:%S')
        self.startTime = time.time()
        self.lastEmitTime = self.startTime
        self.f.write('[{}] {}'.format(self.dtime,"Logging Started\n"))    

    def run(self):
        while self.status == 'run':     
            self.dtime = datetime.now().strftime('%H:%M:%S')
            curTime = time.time()
            fromLast = int(curTime-self.lastEmitTime)
            if fromLast >= self.saveTime:
                self.timeSignal.emit(50)
                self.lastEmitTime = curTime

    def log(self, info):
        log = '[{}] {}\n'.format(self.dtime, info)
        self.f.write(log)
    
    def saveImage(self, name, img):
        cv2.imwrite(name, img.astype("uint8"))

    def __del__(self):
        self.wait()
        print('process thread destruct ...')

    def pause(self):
        self.status = 'pause'
    
    def resume(self):
        self.status = 'run'

    def stop(self):
        print("Userstudy process stopped")
        self.status = 'stop'
        self.f.write('[{}] {}'.format(self.dtime,"Logging Stopped\n"))    
        self.f.close()
    
if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ex = Form()
    sys.exit(app.exec_())