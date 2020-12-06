#importacion de librerias y acceso a archivos 

import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import shutil
from heapq import nsmallest
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE

import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

Base='KS'
bases=['KS']
# Colocar la base de datos que se desea analizar
directorio=str("/home/carlos/Escritorio/ArticuloV/KSC/"+Base)         

n_capas=["","C1","C2","C3","C4"]
for m in range (len(n_capas)):
	capa=n_capas[m]
	print("TLout"+str(capa))
	inicio=datetime.datetime.now()        
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_TLout"+str(capa)+".txt"))
		print("metodo TLout"+str(capa)+" ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Se carga modelo de red 
		from keras.models import load_model
		import h5py

		modelo = str(directorio+"/modelo"+str(capa)+".h5")
		pesos = str(directorio+"/pesos"+str(capa)+".h5")
		model = tf.keras.models.load_model(modelo)
		model.load_weights(pesos)

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		snn_pred = model.predict(datos, batch_size=100)

		#Se aplica TL a los datos
		x=snn_pred
		y=clases

		tomek = TomekLinks(sampling_strategy='all',n_jobs=7)
		x_res, y_res = tomek.fit_resample(x, y)
		print(x_res.shape)
		print(Counter(y_res))
		#Guardar metodo en archivo
		indices=tomek.sample_indices_
		indices.sort()
		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_TLout"+str(capa)+".txt"),'w')

		k=0
		for i in tqdm(range(len(clases))):
			if i == indices[k]and k< (len(indices)-1):
				k=k+1
				for j in range (datos.shape[1]):
					Archivo.write(str(datos[i][j])+" ")
				Archivo.write(str(clases[i]))
				Archivo.write(os.linesep)
		print(len(y_res),k)
		Archivo.close()

	fin=datetime.datetime.now()  
	print(fin-inicio)


	print("ENNout"+str(capa))
	inicio=datetime.datetime.now()      
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ENNout"+str(capa)+".txt"))
		print("metodo ENNout"+str(capa)+" ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Se carga modelo de red 
		from keras.models import load_model
		import h5py

		modelo = str(directorio+"/modelo"+str(capa)+".h5")
		pesos = str(directorio+"/pesos"+str(capa)+".h5")
		model = tf.keras.models.load_model(modelo)
		model.load_weights(pesos)

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		snn_pred = model.predict(datos, batch_size=100)

		#Se aplica edicion
		x=snn_pred
		y=clases

		enn = EditedNearestNeighbours(sampling_strategy='all',n_jobs=7)
		x_res, y_res = enn.fit_resample(x, y)
		print(Counter(y_res))
		indices=enn.sample_indices_
		indices.sort()
		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_ENNout"+str(capa)+".txt"),'w')

		k=0
		for i in tqdm(range(len(clases))):
			if i == indices[k]and k< (len(indices)-1):
				k=k+1
				for j in range (datos.shape[1]):
					Archivo.write(str(datos[i][j])+" ")
				Archivo.write(str(clases[i]))
				Archivo.write(os.linesep)
		print(len(y_res),k)
		Archivo.close()


	fin=datetime.datetime.now()  
	print(fin-inicio)

	print("OSSout"+str(capa))
	inicio=datetime.datetime.now()     
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_OSSout"+str(capa)+".txt"))
		print("metodo OSSout"+str(capa)+" ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Se carga modelo de red 
		from keras.models import load_model
		import h5py

		modelo = str(directorio+"/modelo"+str(capa)+".h5")
		pesos = str(directorio+"/pesos"+str(capa)+".h5")
		model = tf.keras.models.load_model(modelo)
		model.load_weights(pesos)

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		snn_pred = model.predict(datos, batch_size=100)

		#Se aplica OSS a los datos
		x=snn_pred
		y=clases

		oss = OneSidedSelection(sampling_strategy='all',n_jobs=7)
		x_res, y_res= oss.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo
		indices=oss.sample_indices_
		indices.sort()
		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_OSSout"+str(capa)+".txt"),'w')

		k=0
		for i in tqdm(range(len(clases))):
			if i == indices[k] and k< (len(indices)-1):
				k=k+1
				for j in range (datos.shape[1]):
					Archivo.write(str(datos[i][j])+" ")
				Archivo.write(str(clases[i]))
				Archivo.write(os.linesep)
		print(len(y_res),k)
		Archivo.close()

	fin=datetime.datetime.now()  
	print(fin-inicio)




	print("SMOTE-TLout"+str(capa))
	inicio=datetime.datetime.now()      
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-TLout"+str(capa)+".txt"))
		print("metodo SMOTE-TLout"+str(capa)+" ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Se carga modelo de red 
		from keras.models import load_model
		import h5py

		modelo = str(directorio+"/modelo"+str(capa)+"_SMOTE.h5")
		pesos = str(directorio+"/pesos"+str(capa)+"_SMOTE.h5")
		model = tf.keras.models.load_model(modelo)
		model.load_weights(pesos)

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		snn_pred = model.predict(datos, batch_size=100)

		#Se aplica TL a los datos
		x=snn_pred
		y=clases

		tomek = TomekLinks(sampling_strategy='all',n_jobs=7)
		x_res, y_res= tomek.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo
		indices=tomek.sample_indices_
		indices.sort()
		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-TLout"+str(capa)+".txt"),'w')

		k=0
		for i in tqdm(range(len(clases))):
			if i == indices[k]and k< (len(indices)-1):
				k=k+1
				for j in range (datos.shape[1]):
					Archivo.write(str(datos[i][j])+" ")
				Archivo.write(str(clases[i]))
				Archivo.write(os.linesep)
		print(len(y_res),k)
		Archivo.close()
	fin=datetime.datetime.now()  
	print(fin-inicio)


	print("SMOTE-ENNout"+str(capa))
	inicio=datetime.datetime.now()    
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-ENNout"+str(capa)+".txt"))
		print("metodo SMOTE-ENNout"+str(capa)+" ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Se carga modelo de red 
		from keras.models import load_model
		import h5py

		modelo = str(directorio+"/modelo"+str(capa)+"_SMOTE.h5")
		pesos = str(directorio+"/pesos"+str(capa)+"_SMOTE.h5")
		model = tf.keras.models.load_model(modelo)
		model.load_weights(pesos)

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		snn_pred = model.predict(datos, batch_size=100)

		#Se aplica edicion
		x=snn_pred
		y=clases

		enn = EditedNearestNeighbours(sampling_strategy='all',n_jobs=7)
		x_res, y_res= enn.fit_resample(x, y)
		print(Counter(y_res))
		indices=enn.sample_indices_
		indices.sort()
		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-ENNout"+str(capa)+".txt"),'w')
		print(len(indices))
		k=0
		for i in tqdm(range(len(clases))):
			if i == indices[k] and k< (len(indices)-1):
				k=k+1
				for j in range (datos.shape[1]):
					Archivo.write(str(datos[i][j])+" ")
				Archivo.write(str(clases[i]))
				Archivo.write(os.linesep)
		print(len(y_res),k)
		Archivo.close()

	fin=datetime.datetime.now()  
	print(fin-inicio)


	print("SMOTE-OSSout"+str(capa))
	inicio=datetime.datetime.now()      
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-OSSout"+str(capa)+".txt"))
		print("metodo SMOTE-OSSout"+str(capa)+" ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Se carga modelo de red 
		from keras.models import load_model
		import h5py

		modelo = str(directorio+"/modelo"+str(capa)+"_SMOTE.h5")
		pesos = str(directorio+"/pesos"+str(capa)+"_SMOTE.h5")
		model = tf.keras.models.load_model(modelo)
		model.load_weights(pesos)
		#carga=shuffle(carga)
		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE.txt"))

		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		snn_pred = model.predict(datos, batch_size=100)

		#Se aplica OSS a los datos
		x=snn_pred
		y=clases

		oss = OneSidedSelection(sampling_strategy='all',n_jobs=7)
		x_res, y_res = oss.fit_resample(x, y)
		print(Counter(y_res))
		indices=oss.sample_indices_
		#Guardar metodo en archivo
		indices.sort()
		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-OSSout"+str(capa)+".txt"),'w')

		k=0
		for i in tqdm(range(len(clases))):
			if i == indices[k] and k< (len(indices)-1):
				k=k+1
				for j in range (datos.shape[1]):
					Archivo.write(str(datos[i][j])+" ")
				Archivo.write(str(clases[i]))
				Archivo.write(os.linesep)
		print(len(y_res),k)
		Archivo.close()

	fin=datetime.datetime.now()  
	print(fin-inicio)



	print("ROS-TLout"+str(capa))
	inicio=datetime.datetime.now()       
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-TLout"+str(capa)+".txt"))
		print("metodo ROS-TLout"+str(capa)+" ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Se carga modelo de red 
		from keras.models import load_model
		import h5py

		modelo = str(directorio+"/modelo"+str(capa)+"_ROS.h5")
		pesos = str(directorio+"/pesos"+str(capa)+"_ROS.h5")
		model = tf.keras.models.load_model(modelo)
		model.load_weights(pesos)

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		snn_pred = model.predict(datos, batch_size=100)

		#Se aplica TL a los datos
		x=snn_pred
		y=clases

		tomek = TomekLinks(sampling_strategy='all',n_jobs=7)
		x_res, y_res= tomek.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo
		indices=tomek.sample_indices_
		indices.sort()
		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-TLout"+str(capa)+".txt"),'w')

		k=0
		for i in tqdm(range(len(clases))):
			if i == indices[k]and k< (len(indices)-1):
				k=k+1
				for j in range (datos.shape[1]):
					Archivo.write(str(datos[i][j])+" ")
				Archivo.write(str(clases[i]))
				Archivo.write(os.linesep)
		print(len(y_res),k)

	fin=datetime.datetime.now()  
	print(fin-inicio)


	print("ROS-ENNout"+str(capa))
	inicio=datetime.datetime.now()       
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-ENNout"+str(capa)+".txt"))
		print("metodo ROS-ENNout"+str(capa)+" ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Se carga modelo de red 
		from keras.models import load_model
		import h5py

		modelo = str(directorio+"/modelo"+str(capa)+"_ROS.h5")
		pesos = str(directorio+"/pesos"+str(capa)+"_ROS.h5")
		model = tf.keras.models.load_model(modelo)
		model.load_weights(pesos)

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		snn_pred = model.predict(datos, batch_size=100)

		#Se aplica edicion
		x=snn_pred
		y=clases

		enn = EditedNearestNeighbours(sampling_strategy='all',n_jobs=7)
		x_res, y_res= enn.fit_resample(x, y)
		print(Counter(y_res))
		indices=enn.sample_indices_
		indices.sort()
		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-ENNout"+str(capa)+".txt"),'w')

		k=0
		for i in tqdm(range(len(clases))):
			if i == indices[k]and k< (len(indices)-1):
				k=k+1
				for j in range (datos.shape[1]):
					Archivo.write(str(datos[i][j])+" ")
				Archivo.write(str(clases[i]))
				Archivo.write(os.linesep)
		print(len(y_res),k)
		Archivo.close()

	fin=datetime.datetime.now()  
	print(fin-inicio)



	print("ROS-OSSout"+str(capa))
	inicio=datetime.datetime.now()       
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-OSSout"+str(capa)+".txt"))
		print("metodo ROS-OSSout"+str(capa)+" ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Se carga modelo de red 
		from keras.models import load_model
		import h5py

		modelo = str(directorio+"/modelo"+str(capa)+"_ROS.h5")
		pesos = str(directorio+"/pesos"+str(capa)+"_ROS.h5")
		model = tf.keras.models.load_model(modelo)
		model.load_weights(pesos)

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		snn_pred = model.predict(datos, batch_size=100)

		#Se aplica OSS a los datos
		x=snn_pred
		y=clases

		oss = OneSidedSelection(sampling_strategy='all',n_jobs=7)
		x_res, y_res= oss.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo
		indices=oss.sample_indices_
		indices.sort()
		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-OSSout"+str(capa)+".txt"),'w')

		k=0
		for i in tqdm(range(len(clases))):
			if i == indices[k]and k< (len(indices)-1):
				k=k+1
				for j in range (datos.shape[1]):
					Archivo.write(str(datos[i][j])+" ")
				Archivo.write(str(clases[i]))
				Archivo.write(os.linesep)
		print(len(y_res),k)
		Archivo.close()
	fin=datetime.datetime.now()  
	print(fin-inicio)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import random
# Helper libraries
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import optimizers
import matplotlib.pyplot as plt
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


archivos=["_TLout","_ENNout","_OSSout","_SMOTE-TLout","_SMOTE-ENNout","_SMOTE-OSSout","_ROS-TLout","_ROS-ENNout","_ROS-OSSout",
"_TLoutC1","_ENNoutC1","_OSSoutC1","_SMOTE-TLoutC1","_SMOTE-ENNoutC1","_SMOTE-OSSoutC1","_ROS-TLoutC1","_ROS-ENNoutC1","_ROS-OSSoutC1",
"_TLoutC2","_ENNoutC2","_OSSoutC2","_SMOTE-TLoutC2","_SMOTE-ENNoutC2","_SMOTE-OSSoutC2","_ROS-TLoutC2","_ROS-ENNoutC2","_ROS-OSSoutC2",
"_TLoutC3","_ENNoutC3","_OSSoutC3","_SMOTE-TLoutC3","_SMOTE-ENNoutC3","_SMOTE-OSSoutC3","_ROS-TLoutC3","_ROS-ENNoutC3","_ROS-OSSoutC3",
"_TLoutC4","_ENNoutC4","_OSSoutC4","_SMOTE-TLoutC4","_SMOTE-ENNoutC4","_SMOTE-OSSoutC4","_ROS-TLoutC4","_ROS-ENNoutC4","_ROS-OSSoutC4"]


ep=[125]
nc=[14]

def ubicacion(ArchivoEntrenamiento,direccion,Base):
  result=str(direccion+"/"+Base+"_entrenamiento"+ArchivoEntrenamiento+".txt")
  return str(result)

def red(ArchivoEntrenamiento,ArchivoPrueba,n_prueba,Base,numero_clases,Extra,directorio,epo):
  #adquisicion de datos
  datosPrueba=np.loadtxt(ArchivoPrueba)
  datosEntrenamiento=np.loadtxt(str(ArchivoEntrenamiento))
  
  print(datosPrueba.shape)
  print(datosEntrenamiento.shape)

  #se separan los datos por la columna
  columnasD=datosEntrenamiento.shape[1]-1
  columnaC=datosEntrenamiento.shape[1]-1
  x1=datosEntrenamiento[:,0:columnasD]
  y1=datosEntrenamiento[:,columnaC]
  x2=datosPrueba[:,0:columnasD]
  y2=datosPrueba[:,columnaC]
  y1=y1.astype(int)
  y2=y2.astype(int)
  print(len(x1),len(x2),len(y1),len(y2))

  #se define la red, cambia para cada base de datos

  tf.keras.backend.clear_session() 
  capa_entrada= Input(shape=(x1.shape[1],))
  capa1= Dense(50, activation='relu')(capa_entrada)
  capa2= Dense(40, activation='relu')(capa1)
  capa3= Dense(30, activation='relu')(capa2)
  capa4= Dense(20, activation='relu')(capa3)
  capa_salida= Dense(numero_clases, activation='softmax')(capa4)
  salida = Model(inputs=capa_entrada,outputs=capa_salida)

  adam=optimizers.Adam(lr=.001)
  salida.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


  #se inicia entrenamiento, las epocas cambian
  inicio=datetime.datetime.now()
  snn=salida.fit(x1,y1,batch_size=500,nb_epoch=(epo),shuffle=True)

  #plt.plot(snn.history['acc'],label='acc') #Grafica los valores de accuracy por epoca
  fig=plt.plot(snn.history['loss'],label='loss')# Grafica los valores del error cuadratico medio por epoca
  sn.set(font_scale=.8) #for label size
  plt.legend(loc='center right',fontsize='10')
  plt.title(str("Grafica "+str(Extra)+" "+str(n_prueba)+": "+ str(Base)),fontsize='16')
  plt.savefig(str(directorio+"/Grafica"+str(Extra)+str(n_prueba)), bbox_inches='tight')
  plt.close("all")
  #plt.show()
  final=datetime.datetime.now()

  evaluation = salida.evaluate(x1,y1,batch_size=100,verbose=1)#x1,y1,batch_size=100,verbose=1
  snn_pred = salida.predict(x1, batch_size=100) 
  snn_predicted = np.argmax(snn_pred, axis=1)

  #Creamos la matriz de confusión
  snn_cm = confusion_matrix(y1, snn_predicted) 

  #Normalizar matriz de confusión
  snn_cmN= np.zeros((len(snn_cm),len(snn_cm)))
  for i in range(len(snn_cm)):
      total=0
      for k in range(len(snn_cm)):
          total=total+snn_cm[i][k]
          total=total.astype(float)
      for j in range(len(snn_cm)):
          snn_cmN[i][j]=(snn_cm[i][j]/total)

  snn_report = classification_report(y1, snn_predicted,digits=4)
  print(snn_report)
  snn_df_cm = pd.DataFrame(snn_cmN, range(len(Counter(y1))), range(len(Counter(y1)))) ##range son el numero de clases
  #sn.set(font_scale=1.4) #for label size 
  sn.heatmap(snn_df_cm, annot=True) # font size 
  plt.title(str("Matriz de confucion Entrenamiento "+str(Extra)+" "+str(n_prueba)+": "+ str(Base)))
  plt.savefig(str(directorio+"/MatrizEntrenamiento"+str(Extra)+"_"+str(n_prueba)+"_"+str(Base)), bbox_inches='tight')
  plt.close("all")
  #plt.show()
  evaluation = salida.evaluate(x2,y2,batch_size=100,verbose=1)#x1,y1,batch_size=100,verbose=1
  snn_pred = salida.predict(x2, batch_size=100) 
  snn_predicted = np.argmax(snn_pred, axis=1)

  #Creamos la matriz de confusión
  snn_cm = confusion_matrix(y2, snn_predicted) 

  #Normalizar matriz de confusión
  snn_cmN= np.zeros((len(snn_cm),len(snn_cm)))
  for i in range(len(snn_cm)):
      total=0
      for k in range(len(snn_cm)):
          total=total+snn_cm[i][k]
          total=total.astype(float)
      for j in range(len(snn_cm)):
          snn_cmN[i][j]=(snn_cm[i][j]/total)

  # Visualiamos la matriz de confusión 
  snn_df_cm = pd.DataFrame(snn_cmN, range(len(Counter(y2))), range(len(Counter(y2)))) ##range son el numero de clases
  #sn.set(font_scale=1.4) #for label size 
  sn.heatmap(snn_df_cm, annot=True) # font size 
  plt.title(str("Matriz de confucion Prueba "+str(Extra)+" "+str(n_prueba)+": "+ str(Base)))
  plt.savefig(str(directorio+"/MatrizPrueba"+str(Extra)+"_"+str(n_prueba)+"_"+str(Base)), bbox_inches='tight')
  plt.close("all")
  #plt.show()

  snn_report1 = classification_report(y2, snn_predicted,digits=4)
  print(snn_report1)
  
  mult=1
  for i in range(numero_clases):
    mult=mult*snn_cmN[i][i]
  error=np.power(mult,1/numero_clases)
  print(error)
  #se almacena la informacion 
  info=open(str(directorio+'/PruebaDatos'+str(Extra)+" "+str(n_prueba)+'.txt'),'w')
  info.write(str(str(final-inicio)))
  info.write(os.linesep)
  info.write(os.linesep)
  info.write(os.linesep)
  info.write(snn_report)
  info.write(os.linesep)
  info.write(os.linesep)
  info.write(snn_report1)
  info.write(os.linesep)
  info.write(os.linesep)
  info.write(str(error))
  info.close()
  #se guarda el modero

  salida.save(str(directorio+'/Modelo'+str(Extra)+" "+str(n_prueba)+'.h5'))
  salida.save_weights(str(directorio+'/Pesos'+str(Extra)+" "+str(n_prueba)+'.h5'))
  del salida
  return error

#__________________________________________________________________________________

for k in range (len(bases)):
	Base=bases[k]                                               
	directorio=str("/home/carlos/Escritorio/ArticuloV/KSC/"+Base)
	ArchivoTest=str(directorio+"/"+Base+"_test.txt")

	numero_clases=nc[k]
	epocas=ep[k]

	print(Base)
	print(ArchivoTest)


	for i in range(len(archivos)):
		for j in range(1,6):
		    #try:
		      a=ubicacion(archivos[i],directorio,Base)
		      acc=red(a,ArchivoTest,j,Base,numero_clases,archivos[i],directorio,epocas)
		      print(str(archivos[i])+" "+str(j)+"OK")

		    #except:
		    #  print(str(archivos[i])+" "+str(j)+"NOK")