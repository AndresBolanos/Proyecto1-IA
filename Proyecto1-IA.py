# -*- coding: cp1252 -*-

from sklearn.datasets import load_iris
import numpy as np

rango = [[4.3,7.9],[2.0,4.4],[1.0,6.9],[0.1,2.5]]

#Funcion para cargar los datos de Iris
def Load_Irirs_Data():    
    iris = load_iris()
    #print(iris.DESCR)
    #print(iris.data)
    # Arreglos de numpy
    train_data = iris.data
    train_labels = iris.target

    return train_data

#Funcion para cargar los indices de los resultados de Iris
def Load_Iris_Index():
    iris = load_iris()
    train_index = iris.target
    return train_index

#Funcion para cargar los nombres de Iris
def Load_Iris_Names():
    iris = load_iris()
    train_names = iris.target_names
    return train_names

#Funcion para generar el W aleatorio
def Generate_W_Iris(filas, columnas):
    w = np.zeros((filas, columnas))
    #Genero la matriz aleatoria con numeros entre 1 y 255
    #con la cantidad de filas y  columnas que recibe como parametro la funcion
    #w = np.random.uniform(1, 255, (filas, columnas)) #Random con decimales #####CIFAR####
    #retorna el w aleatorio
    for i in range(filas):
        for j in range(columnas):
            w[i][j] = np.random.uniform(rango[j][0], rango[j][1])
    return w 

# Funcion para multiplicar cada elemento de iris
# por el w generado aleatoriamente
def Compare_Iris_Data():
    L = 0; #Contiene la sumatoria de aplicar hinge-loss a cada elemento
    X = Load_Irirs_Data() # Se trae todos los datos de iris
    Index = Load_Iris_Index() # Se trae todos los indices de resultado de iris
    Class = Load_Iris_Names()# Se trae todos los nombres de iris
    W = Generate_W_Iris(len(X[0]), len(Class)+1) #Genera el w aleatorio
    number_Items = len(X) #Largo de los datos
    #Aplicamos la multiplicacion para cada uno de los elementos del data set
    for i in range(number_Items):
        R = np.dot(W, X[i])
        L = L+Hinge_Loss(R,Index[i]) #No se como mandar el correcto
    print("Promedio de perdida "+str(L/number_Items)) 

#Recibe el vector s y la posicion del yi de la clase optimo del arreglo s
def Hinge_Loss(s,yi):
    hinge_loss = 0;
    for i in range(len(s)):
        if (i != yi):
            hinge_loss += np.sum(np.maximum(0,s[i]-s[yi]+1),axis=0)
    return hinge_loss

Compare_Iris_Data()

#Algoritmo gen�tico
#Par�metros:
#   cantidad de generaciones, porcentaje mutaci�n, % cruces
#M�todos para cruce:
#   Cruzar el m�s apto con cada uno de los dem�s
#   El m�s apto con el m�s apto de lo menos aptos y as� sucesivamente
#   Agarrar el porcentaje de cruce y empezar desde el centro para cruzar uno apto con uno no apto
#      y as� hasta llegar al m�s apto con el menos apto (en el caso de 100% de cruce)
