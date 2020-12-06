# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.multiclass import OneVsRestClassifier
import os
from sklearn.metrics import confusion_matrix, classification_report 
import pandas as pd 
import seaborn as sn 
from collections import Counter

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras import optimizers


directorio="/home/carlos/Escritorio/ArticuloV/KSC/KS"
ArchivoPrueba="/home/carlos/Escritorio/ArticuloV/KSC/KS/KS_test.txt"
ArchivoEntrenamiento="/home/carlos/Escritorio/ArticuloV/KSC/KS/KS_entrenamiento_SMOTE.txt"
epocas=125

metodo="_SMOTE"# "", "_ROS", "_SMOTE"


#--------------------------------------------------------------
datosPrueba=np.loadtxt(ArchivoPrueba)
datosEntrenamiento=np.loadtxt(str(ArchivoEntrenamiento))

#print(datosPrueba.shape)
#print(datosEntrenamiento.shape)

columnasD=datosEntrenamiento.shape[1]-1
columnaC=datosEntrenamiento.shape[1]-1
x1=datosEntrenamiento[:,0:columnasD]
y1=datosEntrenamiento[:,columnaC]
x2=datosPrueba[:,0:columnasD]
y2=datosPrueba[:,columnaC]
y1=y1.astype(int)
y2=y2.astype(int)
print(len(x1),len(x2),len(y1),len(y2))


tf.keras.backend.clear_session() 
capa_entrada= Input(shape=(x1.shape[1],))
capa1= Dense(50, activation='relu')(capa_entrada)
capa2= Dense(40, activation='relu')(capa1)
capa3= Dense(30, activation='relu')(capa2)
capa4= Dense(20, activation='relu')(capa3)
capa_salida= Dense(14, activation='softmax')(capa4)

salida = Model(inputs=capa_entrada,outputs=capa_salida)
c4= Model(inputs=capa_entrada, outputs=capa4)
c3= Model(inputs=capa_entrada, outputs=capa3)
c2= Model(inputs=capa_entrada, outputs=capa2)
c1= Model(inputs=capa_entrada, outputs=capa1)

adam=optimizers.Adam(lr=.001)
salida.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
snn=salida.fit(x1,y1,batch_size=500,nb_epoch=(125),shuffle=True)

evaluation = salida.evaluate(x2,y2,batch_size=100,verbose=1)#x1,y1,batch_size=100,verbose=1
print(evaluation)

salida.save(str(directorio+'/modelo'+str(metodo)+'.h5'))
salida.save_weights(str(directorio+'/pesos'+str(metodo)+'.h5'))

c1.save(str(directorio+'/modeloC1'+str(metodo)+'.h5'))
c1.save_weights(str(directorio+'/pesosC1'+str(metodo)+'.h5'))

c2.save(str(directorio+'/modeloC2'+str(metodo)+'.h5'))
c2.save_weights(str(directorio+'/pesosC2'+str(metodo)+'.h5'))

c3.save(str(directorio+'/modeloC3'+str(metodo)+'.h5'))
c3.save_weights(str(directorio+'/pesosC3'+str(metodo)+'.h5'))

c4.save(str(directorio+'/modeloC4'+str(metodo)+'.h5'))
c4.save_weights(str(directorio+'/pesosC4'+str(metodo)+'.h5'))

salida.summary()
c1.summary()
c2.summary()
c3.summary()
c4.summary()

del salida,c1,c2,c3,c4

