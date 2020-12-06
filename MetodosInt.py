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

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import random


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


bases=['KS']

for b in range (len(bases)):
	Base=bases[b]
	# Colocar la base de datos que se desea analizarl
	directorio=str("/home/carlos/Escritorio/ArticuloV/KSC/"+Base)                       # Ruta de la carpeta de la base de datos a anlizar

	print("TL")
	inicio=datetime.datetime.now()      
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_TL.txt"))
		print("metodo TL ya existe")
		
	except:
		#---------------------------------------------------------------------------------

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]


		#Se aplica TL a los datos
		x=datos
		y=clases

		tomek = TomekLinks(sampling_strategy='all',n_jobs=7)
		x_res, y_res = tomek.fit_resample(x, y)

		#Guardar metodo en archivo
		print(Counter(y_res))
		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_TL.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()

	fin=datetime.datetime.now()  
	print(fin-inicio)     


	print("ENN")
	inicio=datetime.datetime.now()        
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ENN.txt"))
		print("metodo ENN ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]


		#Se aplica edicion
		x=datos
		y=clases

		enn = EditedNearestNeighbours(sampling_strategy='all',n_jobs=7)
		x_res, y_res = enn.fit_resample(x, y)
		print(Counter(y_res))

		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_ENN.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()
	fin=datetime.datetime.now()  
	print(fin-inicio)


	print("OSS")
	inicio=datetime.datetime.now()        
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_OSS.txt"))
		print("metodo OSS ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]
		#Se aplica OSS a los datos
		x=datos
		y=clases

		oss = OneSidedSelection(sampling_strategy='all',n_jobs=7)
		x_res, y_res = oss.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_OSS.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()
	fin=datetime.datetime.now()  
	print(fin-inicio)
	    

	print("ROS")
	inicio=datetime.datetime.now()      
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS.txt"))
		print("metodo ROS ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		x=datos
		y=clases

		ros = RandomOverSampler()
		x_res, y_res = ros.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_ROS.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()
	fin=datetime.datetime.now()  
	print(fin-inicio)
	     

	print("SMOTE")
	inicio=datetime.datetime.now()       
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE.txt"))
		print("metodo SMOTE ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]

		x=datos
		y=clases

		smote = SMOTE(n_jobs=7)
		x_res, y_res = smote.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()
	fin=datetime.datetime.now()  
	print(fin-inicio)

	 


	print("SMOTE-TL")
	inicio=datetime.datetime.now()        
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-TL.txt"))
		print("metodo SMOTE-TL ya existe")
		
	except:
	#---------------------------------------------------------------------------------

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]


		#Se aplica TL a los datos
		x=datos
		y=clases

		tomek = TomekLinks(sampling_strategy='all',n_jobs=7)
		x_res, y_res = tomek.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-TL.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()
	fin=datetime.datetime.now()  
	print(fin-inicio)


	print("SMOTE-ENN")
	inicio=datetime.datetime.now()      
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-ENN.txt"))
		print("metodo SMOTE-ENN ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]


		#Se aplica edicion
		x=datos
		y=clases

		enn = EditedNearestNeighbours(sampling_strategy='all',n_jobs=7)
		x_res, y_res = enn.fit_resample(x, y)
		print(Counter(y_res))

		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-ENN.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()
	fin=datetime.datetime.now()  
	print(fin-inicio)


	print("SMOTE-OSS")
	inicio=datetime.datetime.now()       
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-OSS.txt"))
		print("metodo SMOTE-OSS ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]
		#Se aplica OSS a los datos
		x=datos
		y=clases

		oss = OneSidedSelection(sampling_strategy='all',n_jobs=7)
		x_res, y_res = oss.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_SMOTE-OSS.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()
	fin=datetime.datetime.now()  
	print(fin-inicio)


	print("ROS-TL")
	inicio=datetime.datetime.now()      
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-TL.txt"))
		print("metodo ROS-TL ya existe")
		
	except:
	#---------------------------------------------------------------------------------

		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]


		#Se aplica TL a los datos
		x=datos
		y=clases

		tomek = TomekLinks(sampling_strategy='all',n_jobs=7)
		x_res, y_res = tomek.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-TL.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()
	fin=datetime.datetime.now()  
	print(fin-inicio)


	print("ROS-ENN")
	inicio=datetime.datetime.now()    
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-ENN.txt"))
		print("metodo ROS-ENN ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]


		#Se aplica edicion
		x=datos
		y=clases

		enn = EditedNearestNeighbours(sampling_strategy='all',n_jobs=7)
		x_res, y_res = enn.fit_resample(x, y)
		print(Counter(y_res))

		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-ENN.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()

	fin=datetime.datetime.now()  
	print(fin-inicio)


	print("ROS-OSS")
	inicio=datetime.datetime.now()     
	try:
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-OSS.txt"))
		print("metodo ROS-OSS ya existe")
		
	except:
		#---------------------------------------------------------------------------------
		#Leer los valores del archivo de entrenamiento
		carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento_ROS.txt"))
		#carga=shuffle(carga)
		datos=carga[:,0:carga.shape[1]-1]
		clases=carga[:,carga.shape[1]-1]
		#Se aplica OSS a los datos
		x=datos
		y=clases

		oss = OneSidedSelection(sampling_strategy='all',n_jobs=7)
		x_res, y_res = oss.fit_resample(x, y)
		print(Counter(y_res))
		#Guardar metodo en archivo

		Archivo = open(str(directorio+"/"+str(Base)+"_entrenamiento_ROS-OSS.txt"),'w')

		for i in tqdm(range (len(y_res))):
		  for j in range (x_res.shape[1]):
		    Archivo.write(str(x_res[i][j])+" ")
		  Archivo.write(str(y_res[i]))
		  Archivo.write(os.linesep)

		Archivo.close()

	fin=datetime.datetime.now()  
	print(fin-inicio)



# se gerenera vector con las ubicaciones de archivos a ejecutar 

archivos=["","_ROS","_SMOTE","_TL","_ENN","_OSS","_SMOTE-TL","_SMOTE-ENN","_SMOTE-OSS","_ROS-TL","_ROS-ENN","_ROS-OSS"]
ep=[250]
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
