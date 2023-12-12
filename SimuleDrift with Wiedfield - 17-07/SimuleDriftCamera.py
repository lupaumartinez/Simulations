import numpy as np
import time
from datetime import datetime
import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from pyqtgraph.dockarea import Dock, DockArea
from PIL import Image

from scipy import ndimage

import viewbox_tools
import lineprofile

from SimuleDrift import image_NP, video_NP

video_false, x_o, y_o = video_NP(50, 5)

class Frontend(QtGui.QFrame):
    
    liveSignal = pyqtSignal(bool, float)
    imageROI_Signal = pyqtSignal(list, np.ndarray)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setUpGUI()

    def setUpGUI(self):


        self.point_graph_CM = pg.ScatterPlotItem(size=10,
                                                 symbol='+', color='m')    
        
        # taking picture 
        
        time_Label = QtGui.QLabel('Exposure Time (s):')
        self.time_Edit = QtGui.QLineEdit('0.5')

        self.liveview_button = QtGui.QPushButton('LIVEVIEW')
        self.liveview_button.setCheckable(True)
        self.liveview_button.clicked.connect(self.liveview_button_check)
        self.liveview_button.setStyleSheet(
                "QPushButton { background-color: yellow; }"
                "QPushButton:pressed { background-color: blue; }")

        self.ROI_button = QtGui.QPushButton('ROI square')
        self.ROI_button.setCheckable(True)
        self.ROI_button.clicked.connect(self.create_ROI)
        self.ROI_button.setStyleSheet(
                "QPushButton:pressed { background-color: blue; }")

        self.lineprofile_button = QtGui.QPushButton('Line Profile')
        self.lineprofile_button.setCheckable(True)
        self.lineprofile_button.clicked.connect(self.create_line_profile)
        self.lineprofile_button.setStyleSheet(
                "QPushButton:pressed { background-color: blue; }")
              
        
        #Line profile Horizontal:
        
        
        self.lineHorizontal_button = QtGui.QPushButton('Lines Horizontal')
        self.lineHorizontal_button.setCheckable(True)
        self.lineHorizontal_button.clicked.connect(self.create_line_Horizontal)
        self.lineHorizontal_button.setStyleSheet(
                "QPushButton:pressed { background-color: blue; }")
        
        center_row_Label = QtGui.QLabel('Center pixel:')
        self.center_row_Edit = QtGui.QLineEdit('25')
    
        size_spot_Label = QtGui.QLabel('Size Bin:')
        self.size_spot_Edit = QtGui.QLineEdit('25')
    
        self.center_row_Edit.textChanged.connect(self.lines_parameters)
        self.size_spot_Edit.textChanged.connect(self.lines_parameters)
        

        self.camera = QtGui.QWidget()
        camera_parameters_layout = QtGui.QGridLayout()
        self.camera.setLayout(camera_parameters_layout)

        camera_parameters_layout.addWidget(time_Label,              1, 1)
        camera_parameters_layout.addWidget(self.time_Edit,          1, 2)
        camera_parameters_layout.addWidget(self.liveview_button,    2, 1)

        camera_parameters_layout.addWidget(self.ROI_button,         3, 1)
        camera_parameters_layout.addWidget(self.lineprofile_button, 3, 2)
        
    
        camera_parameters_layout.addWidget(size_spot_Label,                    4, 0)
        camera_parameters_layout.addWidget(self.size_spot_Edit,                4, 1)
        camera_parameters_layout.addWidget(center_row_Label,                     5, 0)
        camera_parameters_layout.addWidget(self.center_row_Edit,                 5, 1)
        camera_parameters_layout.addWidget(self.lineHorizontal_button,           6, 1)
        
        
        # image widget layout
        
        imageWidget = pg.GraphicsLayoutWidget()
        imageWidget.setMinimumHeight(100)
        imageWidget.setMinimumWidth(100)
        imageWidget.setAspectLocked(True)
        
        self.vb = imageWidget.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        
        self.axe_x = pg.AxisItem('bottom',  linkView=self.vb, showValues=True)
        self.axe_y = pg.AxisItem('left',  linkView=self.vb, showValues=True)
        
        self.vb.addItem(self.img)
        self.vb.addItem(self.axe_x)
        self.vb.addItem(self.axe_y)
        # set up histogram for the liveview image

        self.hist = pg.HistogramLUTItem(image=self.img)
       # lut = viewbox_tools.generatePgColormap(cmaps.parula)
       # self.hist.gradient.setColorMap(lut)
        self.hist.gradient.loadPreset('thermal')
# 'thermal', 'flame', 'yellowy', 'bipolar', 'spectrum',
# 'cyclic', 'greyclip', 'grey' # Solo son estos

        self.hist.vb.setLimits(yMin=0, yMax=65536)

        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)
        
       

        traceROI_Widget = pg.GraphicsLayoutWidget()  
        self.roi = None
        plot_traceROI =  traceROI_Widget.addPlot(row=1, col=1, title="Trace")
        plot_traceROI.showGrid(x=True, y=True)
        self.curve_ROI = plot_traceROI.plot(open='y')

        self.CM_ROI_Widget = pg.GraphicsLayoutWidget()  
        plot_cmROI =  self.CM_ROI_Widget.addPlot(row=1, col=1, title="CM")
        plot_cmROI.showGrid(x=True, y=True)
        self.curve_CM_x = plot_cmROI.plot(open='y', name='CM x')
        self.curve_CM_y = plot_cmROI.plot(open='y', name='CM y')
        self.curve_xo =  plot_cmROI.plot(open='y', name='xo')
        self.curve_yo =  plot_cmROI.plot(open='y', name='yo')
        
        self.lineplotWidget = lineprofile.linePlotWidget()
        self.lineROI = None
        self.curve_line = self.lineplotWidget.linePlot.plot(open='y')
        
        #docks

        hbox = QtGui.QHBoxLayout(self)
        dockArea = DockArea()
        hbox.addWidget(dockArea)
        self.setLayout(hbox)

        camera_dock = Dock('Parameters')
        camera_dock.addWidget(self.camera)
        dockArea.addDock(camera_dock)
        
        viewbox_dock = Dock('View', size = (70, 70))
        viewbox_dock.addWidget(imageWidget)
        dockArea.addDock(viewbox_dock)
        
        traceROI_dock = Dock('Trace ROI')
        traceROI_dock.addWidget(traceROI_Widget)
        dockArea.addDock(traceROI_dock)
        

    def liveview_button_check(self):
        exposure_time = float(self.time_Edit.text())
        if self.liveview_button.isChecked():
           self.liveSignal.emit(True, exposure_time)
        else:
           self.liveSignal.emit(False, exposure_time)

        
    @pyqtSlot(np.ndarray, float, float)
    def get_image(self, image, xo, yo):

        self.img.setImage(image, autoLevels=True)

        self.xo = xo
        self.yo = yo

        if self.ROI_button.isChecked():
            self.update_ROI(image)
            
        if self.lineprofile_button.isChecked():
            self.update_LINE(image)
            
      #  if self.lineHorizontal_button.isChecked():
       #    self.update_line_Horizontal()
            
            
    def create_line_Horizontal(self):

        if self.lineHorizontal_button.isChecked():
            
            self.center_row = int(self.center_row_Edit.text())
            self.spot_size = int(self.size_spot_Edit.text())

            self.mouse_cursor_x = viewbox_tools.Twolines_fixed(self.vb , self.center_row, self.spot_size)
            self.mouse_cursor_x.show()
       
        else:
            
            self.mouse_cursor_x.hide()
          

    def lines_parameters(self):
        
        self.center_row = int(self.center_row_Edit.text())                                      
        self.spot_size = int(self.size_spot_Edit.text())
        
        #self.lineparametersSignal.emit(center_row, spot_size)
        
   # def update_line_Horizontal(self):
        
      #  x_hu, y_hu = self.mouse_cursor_x.hLine_up.pos()
     #   x_hd, y_hd = self.mouse_cursor_x.hLine_down.pos()
        #x_vu, y_vu = self.mouse_cursor_y.vLine_up.pos()
        #x_vd, y_vd = self.mouse_cursor_y.vLine_down.pos()
        
      #  center_row = (int(y_hu) + int(y_hd) -1)/2
        
   #     print(y_hd, y_hu, center_row)
        
        down_row = self.center_row - int((self.spot_size-1)/2)
        up_row = self.center_row + int((self.spot_size-1)/2)+1
        
        print(down_row, up_row)

        self.mouse_cursor_x.hLine_up.setPos(up_row) 
        self.mouse_cursor_x.hLine_down.setPos(down_row)
        
        x_hu, y_hu = self.mouse_cursor_x.hLine_up.pos()
        x_hd, y_hd = self.mouse_cursor_x.hLine_down.pos()
        
        print(y_hd, y_hu)

    def create_ROI(self):
        
        numberofPixels = 50
        ROIpos = (0.5*numberofPixels-0.5*20, 0.5*numberofPixels-0.5*20)

        if self.ROI_button.isChecked():
            
            self.roi = viewbox_tools.ROI(20, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True)
            
            self.ptr = 0
            self.timeaxis = []
            self.data_ROI = []

            self.dataCM_x = []
            self.dataCM_y = []

            self.data_xo = []
            self.data_yo = []

            self.CM_ROI_Widget.show()
       
        else:
            
            self.CM_ROI_Widget.hide()
            self.point_graph_CM.hide()
            self.roi.hide()
            
        
    
    def update_ROI(self, image):

        
        array_intensity, array_pos = self.roi.getArrayRegion(image, self.img, returnMappedCoords=True)

        x, y = self.roi.pos()

        ROI_pos = [x+0.5, y+0.5]

        self.imageROI_Signal.emit(ROI_pos, array_intensity)
        #self.imageROI_Signal.emit([0,0], image)
        
        mean_intensity = np.round(np.mean(array_intensity), 2)
        step = self.ptr

        self.timeaxis.append(step)

        self.data_ROI.append(mean_intensity)
        self.ptr += 1

        if step < 20:
            self.curve_ROI.setData(self.timeaxis, self.data_ROI,
                           pen=pg.mkPen('g', width=1),
                           shadowPen=pg.mkPen('w', width=3))
        else:
            self.curve_ROI.setData(self.timeaxis[step-20:], self.data_ROI[step-20:],
                           pen=pg.mkPen('g', width=1),
                           shadowPen=pg.mkPen('w', width=3))

        
    def create_line_profile(self):
        
        if self.lineprofile_button.isChecked():

            self.lineROI = pg.LineSegmentROI([[0, 50], [50, 50]], pen='r')
            self.vb.addItem(self.lineROI)
            self.lineplotWidget.show()
                            
        else:

            self.lineplotWidget.hide()
            self.vb.removeItem(self.lineROI)
            
            
    def update_LINE(self, image):
        
        array_intensity = self.lineROI.getArrayRegion(image, self.img)
        
        xmin, ymin = self.lineROI.pos()
        xmax, ymax = self.lineROI.pos() + self.lineROI.size()
                
        array_pos_x = np.linspace(xmin,  xmax, len(array_intensity))
        
        self.curve_line.setData(array_pos_x, array_intensity,
                           pen=pg.mkPen('m', width=1),
                           shadowPen=pg.mkPen('m', width=3)) 
        
    @pyqtSlot(list)
    def get_CMValues(self, data_cm):

        self.point_graph_CM.setData([data_cm[0]], [data_cm[1]])
        self.point_graph_CM.show()
        self.vb.addItem(self.point_graph_CM)

        self.dataCM_x.append(data_cm[0])
        self.dataCM_y.append(data_cm[1])

        self.data_xo.append(self.xo)
        self.data_yo.append(self.yo)

        self.curve_xo.setData(self.timeaxis, self.data_xo,
                           pen=pg.mkPen('m', width=0.5), symbol='o')

        self.curve_yo.setData(self.timeaxis, self.data_yo,
                           pen=pg.mkPen('g', width=0.5), symbol='o')


        self.curve_CM_x.setData(self.timeaxis, self.dataCM_x,
                           pen=pg.mkPen('r', width=1), symbol='o')

        self.curve_CM_y.setData(self.timeaxis, self.dataCM_y,
                           pen=pg.mkPen('b', width=1), symbol='o')


    def closeEvent(self, *args, **kwargs):
        
        super().closeEvent(*args, **kwargs)
        
    def make_connection(self, backend):
        backend.imageSignal.connect(self.get_image)
        backend.CMValuesSignal.connect(self.get_CMValues)
        
        
class Backend(QtCore.QObject):

    imageSignal = pyqtSignal(np.ndarray, float, float)
    CMValuesSignal = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.viewTimer = QtCore.QTimer()
        self.viewTimer.timeout.connect(self.update_view)   
 
     
    @pyqtSlot(bool, float)
    def liveview(self, livebool, exposure_time):
        
        if livebool:
           self.exposure_time = exposure_time    
           self.start_liveview()
        else:
           self.stop_liveview()
    
    def start_liveview(self):

        self.i = 0
  
        #image = video_false[self.i]
        #self.imageSignal.emit(image) 

        self.viewTimer.start(1.5*self.exposure_time*10**3) # ms  , DON'T USE time.sleep() inside the update()

        
    def update_view(self):
        """ Image update while in Liveview mode """

        if self.i < 5:

            image = video_false[self.i]
            xo = x_o[self.i]
            yo = y_o[self.i]
            self.i = self.i + 1
            self.imageSignal.emit(image, xo, yo)

        else:

            image =  video_false[self.i-3]
            xo = x_o[self.i-3]
            yo = y_o[self.i-3]
            self.i = self.i - 3
            self.imageSignal.emit(image, xo, yo)
            

        
    def stop_liveview(self):  

        self.viewTimer.stop()

    
    @pyqtSlot(list, np.ndarray)
    def get_image_ROI(self, pos_ROI, image_ROI):

        self.CMmeasure(pos_ROI, image_ROI)

    def CMmeasure(self, position, imagen):

        Z = imagen  
      
        Zn = Z/max(map(max,Z))  #filtro de %70
        for i in range(len(Z[:,1])):
            for j in range (len(Z[1,:])):
                if Zn[i,j] < 0.7:
                    Zn[i,j] = 0
                                               
        xcm, ycm = ndimage.measurements.center_of_mass(Z)

        x_cm = position[0] + xcm
        y_cm = position[1] + ycm

        self.CMValuesSignal.emit([x_cm, y_cm])
        
        return xcm, ycm
    
    def make_connection(self, frontend):

        frontend.liveSignal.connect(self.liveview)
        frontend.imageROI_Signal.connect(self.get_image_ROI)

if __name__ == '__main__':

    app = QtGui.QApplication([])

    gui = Frontend()   
    worker = Backend()

    worker.make_connection(gui)
    gui.make_connection(worker)

    cameraThread = QtCore.QThread()
    worker.moveToThread(cameraThread)
    worker.viewTimer.moveToThread(cameraThread)
    cameraThread.start()

    gui.show()
    app.exec_()