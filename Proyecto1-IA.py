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


#Funcion para cruzar en el set de datos de Iris
#Unicamente esta cruzando dos filas, las que tienen mas loss pro se encicla
def Cruce1(porcentaje,W,Loss):
    print(Loss)
    Cantidad = len(W) * porcentaje
    pos1 = np.argmax(Loss)
    Loss[pos1] = 0
    pos2 = np.argmax(Loss)
    aux = np.copy(W[pos1])
    W[pos1] = W[pos2]
    W[pos2] = aux
    
    return W

#Funcion para mutar una fila del set de iris
#Muta la fila con mas Loss
def Mutacion1(porcentaje,W,Loss):
    cantidad_mutaciones = round(len(W[0])*porcentaje)
    pos1 = np.argmax(Loss) #posicion del menos apto    #HACIENDO USO DE ESTA LINEA SE MUTA SOLO EL QUE TIENE MAS LOSS --#1
    for i in range(int(cantidad_mutaciones)):
        #pos1 = np.random.randint(0,len(W)) #HACIENDO USO DE ESTA LINEA SE MUTA TODO OSEA CADA LINEA DE W --#25
        pos = np.random.randint(0,len(W[0]))#Ver en que atributo se va a mutar
        nuevo  = np.random.uniform(rango[pos][0], rango[pos][1]) #Porcentaje de la mutacion
        W.itemset((pos1, pos), nuevo)
    return W

    
# Funcion para multiplicar cada elemento de iris
# por el w generado aleatoriamente
def Compare_Iris_Data():
    Lista_Loss = [] #Guarda el loss para cada clase
    L = 0; #Contiene la sumatoria de aplicar hinge-loss a cada elemento
    X = Load_Irirs_Data() # Se trae todos los datos de iris
    Index = Load_Iris_Index() # Se trae todos los indices de resultado de iris
    Class = Load_Iris_Names()# Se trae todos los nombres de iris
    W = Generate_W_Iris(len(X[0]), len(Class)+1) #Genera el w aleatorio
    number_Items = len(X) #Largo de los datos
    #Aplicamos la multiplicacion para cada uno de los elementos del data set
    cont = 0
    promedio = 0
    for i in range(1000):
        for i in range(number_Items):
            R = np.dot(W, X[i])
            if (cont < 49):
                L = L+Hinge_Loss(R,Index[i])
                cont+=1
            else:
                L = L+Hinge_Loss(R,Index[i])
                Lista_Loss.append(L)
                L = 0
                cont = 0
        #print(Lista_Loss)
        W = Mutacion1(1.0,W,Lista_Loss) #0.90 o mas es un buen valor
        promedio = 0
        for i in range(len(Lista_Loss)):
            promedio += Lista_Loss[i]
        promedio = promedio/len(Lista_Loss)
        Lista_Loss = []
    print("Promedio "+str(promedio))
    print(W)

    
#Recibe el vector s y la posicion del yi de la clase optimo del arreglo s
def Hinge_Loss(s,yi):
    hinge_loss = 0;
    for i in range(len(s)):
        if (i != yi):
            hinge_loss += np.sum(np.maximum(0,s[i]-s[yi]+1),axis=0)
    return hinge_loss

Compare_Iris_Data()

#Algoritmo genético
#Parámetros:
#   cantidad de generaciones, porcentaje mutación, % cruces
#Métodos para cruce:
#   Cruzar el más apto con cada uno de los demás
#   El más apto con el más apto de lo menos aptos y así sucesivamente
#   Agarrar el porcentaje de cruce y empezar desde el centro para cruzar uno apto con uno no apto
#      y así hasta llegar al más apto con el menos apto (en el caso de 100% de cruce)












