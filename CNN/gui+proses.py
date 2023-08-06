import sys 
import cv2 
import os 
import numpy as np 
import pandas as pd
import source 
import imutils
import tensorflow as tf

import gui_nanas
from gui_nanas import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget
from PyQt5.QtGui import QIcon,QPixmap,QImage,QColor,QImageWriter
from PyQt5.QtCore import QTimer 
from PyQt5.uic import loadUi

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.ui=Ui_MainWindow()
if __name__== '__main__':
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Kontrol"))
        self.pushButton_9.setText(_translate("MainWindow", "Start"))
        self.pushButton_10.setText(_translate("MainWindow", "Capture"))
        self.pushButton_11.setText(_translate("MainWindow", "Proses"))
        self.pushButton_12.setText(_translate("MainWindow", "Save"))
        self.pushButton_17.setText(_translate("MainWindow", "Exit"))
    def setupUi(self, QMainWindow):
        self.ui.setupUi(self)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.ViewCam)
        self.ui.Start_pushButton_9.clicked.connect(self.KontrolTimer)
        self.ui.Capture_pushButton_10.clicked.connect(self.Capture)
        self.ui.Proses_pushButton_11.clicked.connect(self.Proses)
        self.ui.Save_pushButton_12.clicked.connect(self.Save)
        self.ui.Exit_pushButon_17.clicked.connect(self.Exit)
#memulai (Start)   
def ViewCam(self):
    ret,img=self.cap.read()
    img=cv2.cvtColor(img.cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(261,191),interpolation=cv2.INTER_AREA)
    height,width,channel=img.shape
    step=channel*width
    img1=QImage(img.data,width,height,step,QImage.Format_RGB888)
    self.ui.kamera.setPixmap(QPixmap.fromImage(img1))
     
def kontrolTimer(self):
    if not self.timer.isActive():
        self.cap=cv2.VideoCapture(0)
        self.timer.stert(20)
    else:
        self.timer.stop()
        self.cap.release()
        self.ui.Start_pushButton.setText("Start")
#capture
def Capture(self):
    ret,img=self.cap.read()
    rows,cols,_=img.shape
    up=(int(cols/6),int(rows/6))
    bottom=(int(cols*5/6),int(rows*5/6))
    img=img[up[1]:bottom[1],up[0]:bottom[0]]
    self.imgg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(self.imgg,(261,191),interpolation=cv2.INTER_AREA)
    height,width,channel=img.shape
    step=channel*width
    
    self.img1=QImage(img.data,width,height,step,QImage.Format_RGB888)
    self.ui.capture.setPixmap(QPixmap.fromImage(self.img1))

#PROSESsss    
#pelatihan cnn dan jst- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def Proses(self):
    import pandas as pd
    import cv2
from keras.preprocessing.image import ImageDataGenerator  #from tensorflow.keras_preprocessing.image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
 
# Lokasi dataset sesuai dengan struktur folder
path_to_dataset = ('train.csv')
path_to_dataset = ('validation.csv')
path_to_dataset = ('test.csv')

dataset = pd.read_csv(path_to_dataset)
print(dataset.head())
# Data augmentation untuk data pelatihan
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Rescale data validasi dan pengujian
validation_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)
# Ukuran batch dan dimensi gambar
batch_size = 32
image_width, image_height = 150, 150
# Memuat data pelatihan, validasi, dan pengujian dari folder
train_generator = train_datagen.flow_from_directory(
    path_to_dataset,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode="categorical"
)
validation_generator = validation_datagen.flow_from_directory(
    path_to_dataset,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    path_to_dataset,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)
# Membuat model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 kelas untuk kematangan buah nanas
])
# Kompilasi model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Melatih model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
# Prediksi pada contoh gambar baru
# Replace 'path_to_new_image' with the actual path to the new image
new_image = tf.keras.preprocessing.image.load_img('path_to_new_image', target_size=(image_width, image_height))
new_image = tf.keras.preprocessing.image.img_to_array(new_image)
new_image = new_image.reshape((1,) + new_image.shape)
new_image /= 255.0

prediction = model.predict(new_image)
print(f'Prediction: {prediction}')

#save
def save(self):
    if self.img1 is None:
        QMessageBox.about(self,"Peringatan!","Gambar belum di Capture")
    else:
        files_types="JPG(*.jpg);;PNG(*.png);;JPEG(*,jpeg);;TIFF(*.tiff);;BMP(*,bmp)"
        new_omg_path,_=QMainWindow.getSaveFileName(self,'Save Image','./',files_types)
        img=self.imgg
        if new_img_path!=":":new_img_path=str(new_img_path)
        nameFile=new_img_path.split(".")[0]
        ekstensiFile=new_img_path.split(".")[-1]
        cv2.imwrite(nameFile+'_Capture.' +ekstensiFile, cv2.cvtColor(self.imgg, cv2.COLOR_BGR2RGB))
        
    if ekstensiFile=='jpg' or ekstensiFile=='png' or ekstensiFile=='jpeg' or ekstensiFile=='bmp' or ekstensiFile=='tiff': QMessageBox.about(self, "Pemberitahuan!", "Gambar Tersimpan")
    
#exit
def exit(self):
    qm=QMessageBox.question(self,'Confirm Quit',"Apakah Anda Ingin Keluar?", QMessageBox.Yes | QMessageBox.No) 
    if QMessageBox.Yes:
        self.close()
        return True
    else:
        return False
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())