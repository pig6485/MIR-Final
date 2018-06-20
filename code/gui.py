import sys
 
from PyQt5.QtWidgets import (QApplication, 
                             QMainWindow, 
                             QMenu, 
                             QVBoxLayout, 
                             QGridLayout,
                             QSpacerItem,
                             QSizePolicy,
                             QMessageBox, 
                             QWidget, 
                             QPushButton,
                             QDialog,
                             QDialogButtonBox,
                             QLabel,
                             QLineEdit,
                             QSpinBox,
                             QFileDialog,
                             QProgressBar)
from PyQt5.QtCore import QObject, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import feature as ft
import librosa.core as lib
import notetrack as nt
import random
 
class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'MIR Final Demo'
        self.width = 640
        self.height = 450
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.vocalMax = 60
        self.vocalMin = 50
        
        self.move(100, 100)
        self.songfile = '../data/13_LeadVox.wav'
        self.m = PlotCanvas(self, width=5, height=4, songfile = self.songfile,
                            vocalRange = [self.vocalMin, self.vocalMax])
        self.m.move(0,70)
        
        self.addToolBar(NavigationToolbar(self.m, self))
        
        button = QPushButton('Change Plot', self)
        button.setToolTip('This s an example button')
        button.clicked.connect(self.m.plot)
        button.move(500,20)
        button.resize(140,20)
        
        button = QPushButton('Input', self)
        button.setToolTip('This s an example button')
        button.clicked.connect(self.getInteger)
        button.move(500,40)
        button.resize(140,20)
        
        button = QPushButton('Open File', self)
        button.setToolTip('This s an example button')
        button.clicked.connect(self.getSongName)
        button.move(500,60)
        button.resize(140,20)
        
        self.label = QLabel(self)
        self.label.move(500, 80)
        self.label.resize(140, 40)
        self.label.show()
        
        self.songNameLabel = QLabel(self)
        newfont = QFont("Times", 12, QFont.Normal)
        self.songNameLabel.setFont(newfont)
        self.songNameLabel.move(10, 35)
        self.songNameLabel.resize(500, 40)
        self.songNameLabel.setWordWrap(True)
        self.songNameLabel.setAlignment(Qt.AlignTop)
        self.songNameLabel.show()
        
        self.songScoreLabel = QLabel(self)
        newfont = QFont("Times", 16, QFont.Bold) 
        self.songScoreLabel.setFont(newfont)
        self.songScoreLabel.move(500, 130)
        self.songScoreLabel.resize(140, 30)
        self.songScoreLabel.show()
        
        
        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0,1)
        self.progressBar.move(self.width - 140, self.height - 20)
        self.progressBar.resize(140, 20)
        self.progressBar.show()
        
        self.show()
        self.updateLabel()
        
    def getVocalValue(self):
        return self.vocalMin, self.vocalMax
        
        
    def getInteger(self):
        dialog = Dialog(parent=self)
        if dialog.exec_():
           #todo
           self.vocalMax, self.vocalMin = dialog.value()
           self.updateLabel()
           self.m.computeSongFeatureScore(self.vocalMin, self.vocalMax)
           self.m.plot()
        dialog.destroy()
        

    def getSongName(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Wave Files (*.wave)", options=options)
        oldSongFile = self.songfile
        if fileName:
            self.songfile = fileName
            print('load new song:', fileName)
        

            self.updateLoadingLabel()
            ok = self.m.loadSong(self.songfile)
            #task = loadSongTaskThread()
            if ok:
                self.m.computeSongFeatureScore(self.vocalMin, self.vocalMax)
                self.updateLabel()
                self.m.plot()
            else:
                self.songfile = oldSongFile
                self.updateLabel()
        else:
             QMessageBox.critical(self, 'File name error', 
                                    u'沒有此檔案或檔名錯誤!', 
                                    QMessageBox.Ok, QMessageBox.Ok)
             self.songfile = oldSongfile
             return False
    def updateLabel(self):
        self.label.setText("Vocal Range: \nmin:{min}, max:{max}".format(min = self.vocalMin,
                                         max = self.vocalMax))
        self.songNameLabel.setText("Song: \n{}".format(self.songfile))
        self.songScoreLabel.setText("Score: {s:2.2f}".format(s=self.m.data[-1]))
        
    def updateLoadingLabel(self):
        print('update loading label')
        self.songNameLabel.setText("Loading Song: \n{}".format(self.songfile))
 
    def updateProgressBar(self, text, value):
        self.progress_bar.setValue(value)
 
class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100, songfile = None,
                 vocalRange = None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.vocalRange = vocalRange
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.songfile = songfile
        self.songY, self.songSr = lib.load(self.songfile)
        self.computeSongFeatureScore(self.vocalRange[0], self.vocalRange[1])
        self.plot()
 
    def plot(self):
        self.axes.clear()
        container = self.axes.bar(range(len(self.data)), 
                                  self.data, 
                                  tick_label=self.cat)
        container[-1].set_facecolor('red')
        print(len(container))
        self.axes.set_title('Score Graph')
        self.draw()
        
    def loadSong(self, songfile):
        if songfile != self.songfile:
            oldSongFile = self.songfile
            self.songfile = songfile
            print('load song:{}'.format(self.songfile))
            try:
                self.songY, self.songSr = lib.load(songfile)
            except Exception:
                QMessageBox.critical(self, 'File loading failed!', 
                                    u'載入檔案失敗', 
                                    QMessageBox.Ok, QMessageBox.Ok)
                self.songfile = oldSongfile
                return False
            
            return True
        else:
            buttonReply = QMessageBox.warning(self, 'File Same message', 
                                               u'選擇的檔案與先前相同!', 
                                               QMessageBox.Ok, QMessageBox.Ok)
            return False
            
    def computeSongFeatureScore(self, vocalMin, vocalMax):
        phraseScore = ft.phraseDetect(self.songY, self.songSr)
        notes = notes = np.round(nt.midfilter(nt.notetrack(self.songY), 
                                              window_size = 5))
        intervalScore = ft.intervalDetect(notes)
        userNotes = np.zeros(80)
        print(type(self.parent))
        userNotes[vocalMin:vocalMax] = 1
        vocalRangeScore = ft.vocalRangeDetect(notes, userNotes)
        noteChangeScore = ft.noteChange(notes, sr = self.songSr)
        self.data = [phraseScore, 
                     intervalScore, 
                     vocalRangeScore, 
                     noteChangeScore]
        totalScore = np.mean(self.data)
        self.data.append(totalScore)
        
        self.cat = ['Phrase',
                    'Interval',
                    'Range',
                    'Note Change/s',
                    'Score']
        
        
class Dialog(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.resize(240, 200)
        self.title = 'Vocal Input Dialog'
        # 表格布局，用来布局QLabel和QLineEdit及QSpinBox
        grid = QGridLayout()
        grid.addWidget(QLabel(u'最高音', parent=self), 0, 0, 1, 1)
        self.hiVocal = QSpinBox(parent=self)
        grid.addWidget(self.hiVocal, 0, 1, 1, 1)
        grid.addWidget(QLabel(u'最低音', parent=self), 1, 0, 1, 1)
        self.lwVocal = QSpinBox(parent=self)
        grid.addWidget(self.lwVocal, 1, 1, 1, 1)
        # 创建ButtonBox，用户确定和取消
        buttonBox = QDialogButtonBox(parent=self)
        buttonBox.setOrientation(Qt.Horizontal) # 设置为水平方向
        buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok) # 确定和取消两个按钮
        # 连接信号和槽
        buttonBox.accepted.connect(self.accept) # 确定
        buttonBox.rejected.connect(self.reject) # 取消
        # 垂直布局，布局表格及按钮
        layout = QVBoxLayout()
        # 加入前面创建的表格布局
        layout.addLayout(grid)
        # 放一个间隔对象美化布局
        spacerItem = QSpacerItem(20, 48, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacerItem)
        # ButtonBox
        layout.addWidget(buttonBox)
        self.setLayout(layout)
    def value(self):
        return self.hiVocal.value(), self.lwVocal.value()
        

class MyCustomWidget(QWidget):

    def __init__(self, parent=None):
        super(MyCustomWidget, self).__init__(parent)
        layout = QVBoxLayout(self)

        # Create a progress bar and a button and add them to the main layout
        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0,1)
        layout.addWidget(self.progressBar)
        button = QPushButton("Start", self)
        layout.addWidget(button)      

        button.clicked.connect(self.onStart)

        self.myLongTask = TaskThread()
        self.myLongTask.taskFinished.connect(self.onFinished)

    def onStart(self): 
        self.progressBar.setRange(0,0)
        self.myLongTask.start()

    def onFinished(self):
        # Stop the pulsation
        self.progressBar.setRange(0,1)


class loadSongTaskThread(QThread):
    def __init__(self, songfile):
        super.__init__(self)
        self.songfile = songfile

    
    taskFinished = pyqtSignal(list, float)
    def run(self):
        songY, songSr = lib.load(songfile)
        self.taskFinished.emit(songY, songSr)

class computeSongFeatureScoreTaskThread(QThread):
    
    def __init__(self, y, sr, vocalMin, vocalMax):
        super.__init__(self)
        self.songY = y
        self.songSr = sr
        self.vocalMin = vocalMin
        self.vocalMax = vocalMax
    
    taskFinished = pyqtSignal(list, list)
    updateValue = pyqtSignal(float)
    def run(self):
        phraseScore = ft.phraseDetect(self.songY, self.songSr)
        self.updateValue(float).emit(0.2)
        notes = notes = np.round(nt.midfilter(nt.notetrack(self.songY), 
                                              window_size = 5))
        self.updateValue(float).emit(0.4)
        intervalScore = ft.intervalDetect(notes)
        self.updateValue(float).emit(0.6)
        userNotes = np.zeros(80)
        userNotes[self.vocalMin:self.vocalMax] = 1
        vocalRangeScore = ft.vocalRangeDetect(notes, userNotes)
        self.updateValue(float).emit(0.8)
        noteChangeScore = ft.noteChange(notes, sr = self.songSr)
        self.updateValue(float).emit(0.9)
        data = [phraseScore, 
                     intervalScore, 
                     vocalRangeScore, 
                     noteChangeScore]
        totalScore = np.mean(self.data)
        data.append(totalScore)
        
        cat = ['Phrase',
                    'Interval',
                    'Range',
                    'Note Change/s',
                    'Score']
        self.taskFinished.emit(data, cat)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())