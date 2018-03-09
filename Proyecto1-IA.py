# -*- coding: cp1252 -*-

from sklearn.datasets import load_iris
import numpy as np
import pickle
import collections
import random

rango = [[4.3,7.9],[2.0,4.4],[1.0,6.9],[0.1,2.5]]
####################################### LOAD IRIS DATA #######################################
#Funcion para cargar los datos de Iris
def Load_Irirs_Data():    
    iris = load_iris()
    #print(iris.DESCR)
    train_data = iris.data
    train_data = np.resize(train_data,(len(train_data),len(train_data[0])+1))
    for i in range(len(train_data)):
        train_data.itemset((i, 4), 1.0)
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


#################################### LOAD CFAR-10 DATA #######################################
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#Carga los datos entrenamiento de CFAR
def Load_CFAR_Data(labels):
    s = unpickle(".\\cifar-10-batches-py\\data_batch_1")
    images = s[b'data']
    
    cont = 2
    while cont <= 5:    
        s = unpickle(".\\cifar-10-batches-py\\data_batch_"+str(cont))
        images = np.append(images,s[b'data'],0)
        cont+=1

    #   
    mylist_data = [] #Guarda los datos con clase menor o igual a las tres primeras clases
    for i in range(len(images)):
        if (labels[i] <= 3):
            vector = []
            for j in range(len(images[i])/3):
                r = images[i][j]
                g = images[i][j+1023]
                b = images[i][j+2046]
                vector += [int((int(r)+int(g)+int(b))/3)]
            vector += [1]
            mylist_data.append(vector)
    final_data = np.array(mylist_data)
    return final_data

#Carga los datos de prueba de CFAR
def Load_CFAR_Data_Test(labels_test):
    s = unpickle(".\\cifar-10-batches-py\\test_batch")
    images = s[b'data']
 
    mylist_data = [] #Guarda los datos con clase menor o igual a las tres primeras clases
    for i in range(len(images)):
        if (labels_test[i] <= 3):
            vector = []
            for j in range(len(images[i])/3):
                r = images[i][j]
                g = images[i][j+1023]
                b = images[i][j+2046]
                vector += [int((int(r)+int(g)+int(b))/3)]
            vector += [1]
            mylist_data.append(vector)
    final_data = np.array(mylist_data)
    return final_data

#Esta funcion retorna un arreglo con las posiciones correctas de la clases que se van a usar
def get_Labels(labels):
    mylist_labels = [] #Guarda los labels con clase menor o igual a las 4 primeras clases
    for i in range(len(labels)):
        if (labels[i] <= 3):
            mylist_labels.append(labels[i])
    final_labels = np.array(mylist_labels)
    return final_labels

#Carga los labels de entrenamiento de CFAR
def Load_CFAR_Labels():
    s = unpickle(".\\cifar-10-batches-py\\data_batch_1")
    labels = np.asarray(s[b'labels'])
    
    cont = 2
    while cont <= 5:    
        s = unpickle(".\\cifar-10-batches-py\\data_batch_"+str(cont))
        labels = np.append(labels,np.asarray(s[b'labels']),0)
        cont+=1
    return labels

#Carga los labels de prueba de CFAR
def Load_CFAR_Labels_Test():
    s = unpickle(".\\cifar-10-batches-py\\test_batch")
    labels = np.asarray(s[b'labels'])

    return labels

#################################### RANDOM W GENERATION #######################################
#Funcion para generar el W aleatorio en CFAR-10 y Iris con una distribucion normal
def Generate_W(filas, columnas):
    #Genero la matriz aleatoria con numeros entre 1 y 255
    #con la cantidad de filas y  columnas que recibe como parametro la funcion
    #w = np.random.randint(1, 255, (filas, columnas)) #Random con decimales #####CIFAR###
    mu, sigma = 50,22 # mean and standard deviation
    W = []
    for i in range (filas):
        W.append(np.absolute(np.random.normal(mu, sigma, columnas).tolist()))
    #print("Largo de filas de W "+str(len(W)))
    #print("Largo de columnas de W "+str(len(W[0])))   
    W = np.array(W)
    return W


#################################### MIX W #######################################
#Funcion para cruzar en el set de datos de Iris
#Unicamente esta cruzando dos filas, las que tienen mas loss pro se encicla


#################################### MUTATION W #######################################


#Funcion de mutacion 1
#Selecciona con base al porcentaje de mutacion una determinada cantidad de W
#Genera un nuevo W con valores random de posiciones random segun la cantidad q se indica en el procentaje de mutacion 2
#Ese nuevo W se cambia por el que tenia el Loss mas bajo
def Mutacion_1(Lista_W,Lista_Indices,Lista_Loss,mutacion_1, mutacion_2):
    cantidad_W = int(round(len(Lista_W) * mutacion_1)) #Calcula la cantidad de W a escoger segun el procentaje
    cantidad_Genes = int(round(len(Lista_W[0][0])) * mutacion_2) #Indica la cantidad de genes que se van a mutar dentro de la W
    posicion_1 = ((len(Lista_Indices)/2) -1)+len(Lista_Indices)%2 #Seleccion los masomenos aptos para mutar
    posicion_2 = posicion_1 +1 #Escoge el siguiente mas apto
    for c in range(cantidad_W):
        W1 = Lista_W[Lista_Indices[posicion_1]]
        W2 = Lista_W[Lista_Indices[posicion_2]]
        nueva_Matriz = np.copy(W2)
        for g in range(cantidad_Genes): #De acuerdo al porcenta de mutacion 2
            i = random.randint(0,len(W1)-1)
            j = random.randint(0,len(W1[0])-1)
            nueva_Matriz[i][j] = random.uniform(W1[i][j],W2[i][j]) #Asigna el valor del random del rango
        Lista_W[Lista_Indices[posicion_2]] = nueva_Matriz
        #Movemos los pivotes en ambos extremos para la siguiente iteracion
        posicion_1 -= 1
        posicion_2 += 1
    return Lista_W 

#Forma de mutacion 2
#Selecciona con base al porcentaje de mutacion una determinada cantidad de W
#Genera un nuevo W con el procentaje de mutacion 2 en cada fila de W
#Aplica el porcentaje a cada fila de W
#Mantiene el mas apto
def Mutacion_2(Lista_W,Lista_Indices,Lista_Loss,mutacion_1, mutacion_2):
    cantidad_W = int(round(len(Lista_W) * mutacion_1)) #Calcula la cantidad de W a escoger segun el procentaje
    cantidad_Genes = int(round(len(Lista_W[0][0])) * mutacion_2) #Indica la cantidad de genes que se van a mutar dentro de la W
    posicion_1 = ((len(Lista_Indices)/2) -1)+len(Lista_Indices)%2 #Seleccion los masomenos aptos para mutar
    posicion_2 = posicion_1 +1 #Escoge el siguiente mas apto
    for c in range(cantidad_W):
        W1 = Lista_W[Lista_Indices[posicion_1]]
        W2 = Lista_W[Lista_Indices[posicion_2]]
        nueva_Matriz = np.copy(W2)
        for f in range(len(W1)): #Recorre cada fila de W
            for g in range(cantidad_Genes):
                j = random.randint(0,len(W1[0])-1)
                nueva_Matriz[f][j] = random.uniform(W1[f][j],W2[f][j]) #Asigna el valor del random del rango
        Lista_W[Lista_Indices[posicion_2]] = nueva_Matriz
        posicion_1 -= 1
        posicion_2 += 1 #Movemos los pivotes en ambos extremos para la siguiente iteracion
    return Lista_W  


#################################### COMPARISON DATA FUNCTIONS #######################################    

#Funcion para calcular el loss de cada clase y el total de W
#etorna un arreglo con el loss de cada clase y en la ultima posicion el loss de W
def Calculo_Loss(W,X,Labels): 
    cont = len(collections.Counter(Labels).items()) #COntar la cantidad de clases que hay
    Lista_Loss = np.zeros((cont+1), dtype=int) #Genera un arreglo de ceros del largos de las clases
    for i in range(len(X)):
        R = np.dot(W, X[i]) #Multiplicacion de W por cada imagen
        L = Hinge_Loss(R,Labels[i]) #Loss para el vector solucion
        Lista_Loss[Labels[i]] = Lista_Loss[Labels[i]]+ L #Guarda el loss en la respectiva posicion  
    Lista_Loss[cont] = np.sum(Lista_Loss) #Hace la sumatoria de todos los loss y lo pone en la ultima posicion
    return Lista_Loss

def Compare_Iris_Data(k,mutacion_1,mutacion_2,cruce):
    Lista_Loss = [] #Guarda el loss para cada clase
    L = 0; #Contiene la sumatoria de aplicar hinge-loss a cada elemento
    X = Load_Irirs_Data() # Se trae todos los datos de iris
    Index = Load_Iris_Index() # Se trae todos los indices de resultado de iris
    Class = Load_Iris_Names()# Se trae todos los nombres de iris
    number_Items = len(X) #Largo de los datos
    Lista_W = []
    Lista_Loss = []
    Lista_Indices = []
    for i in range(k):
        Lista_W.append(Generate_W(len(Class),len(X[0])))#Genera el w aleatorio ##Automaticamente en la funcion que genera el +1 del Bias Trick
        Lista_Loss.append(Calculo_Loss(Lista_W[i],X,Index)) #Guarda la lista de loss de clase y de W para cada W generado
        Lista_Indices = Insertar_Indices(Lista_Indices,i,Lista_Loss)

    Mutacion_2(Lista_W,Lista_Indices,Lista_Loss,mutacion_1, mutacion_2)
    #print(Lista_Indices)
    #print(Lista_Loss)

def Insertar_Indices(Lista_Indices,i,Lista_Loss):
    if Lista_Indices == []:
        Lista_Indices.append(i)
        return Lista_Indices
    else:
        for j in range(len(Lista_Indices)):
            if Lista_Loss[Lista_Indices[j]][len(Lista_Loss[0])-1] > Lista_Loss[i][len(Lista_Loss[0])-1]:
                Lista_Indices = Lista_Indices[:j]+[i]+Lista_Indices[j:]
                return Lista_Indices
                
            else:
                if j == len(Lista_Indices)-1:
                    Lista_Indices = Lista_Indices+[i]
                    return Lista_Indices

                
    
# Funcion para multiplicar cada elemento de cfar
# por el w generado aleatoriamente
def Compare_CFAR_Data():
    Labels = Load_CFAR_Labels()
    X = Load_CFAR_Data(Labels); # Se trae todos los datos de cfar-10
    Labels_Test = Load_CFAR_Labels_Test()
    X_Test = Load_CFAR_Data_Test(Labels_Test)
    Labels = get_Labels(Labels)
    Labels_Test = get_Labels(Labels_Test)
    print("Entrenamiento largo vector "+str(len(X[0])))
    print("Labels  "+str(len(Labels)))
    print("Prueba largo vector "+str(len(X_Test[0])))
    print("Labels  "+str(len(Labels_Test)))
    #Pongo 4 porque son solo 4 clases hay q mapearlo con el dato bien
    W = Generate_W(4,len(X[0])) #Genera el w aleatorio ##Automaticamente en la funcion que genera el +1 del Bias Trick
    number_Items = len(X) #Largo de los datos
    L = 0; #Contiene la sumatoria de aplicar hinge-loss a cada elemento
    for i in range(1000): #El valor 1000 es la cantidad de iteraciones q permite hacer  es un hiperparametro
        for i in range(number_Items):
             R = np.dot(W, X[i])
             L = Hinge_Loss(R,Labels[i])
             #Hay q ver para promediar cada clase al igual que en Iris
             
    
#################################### HINGE LOSS FUNCTION #######################################    
#Recibe el vector s y la posicion del yi de la clase optimo del arreglo s
def Hinge_Loss(s,yi):
    hinge_loss = 0;
    for i in range(len(s)):
        if (i != yi):
            hinge_loss += np.sum(np.maximum(0,s[i]-s[yi]+1),axis=0)
    return hinge_loss

Compare_Iris_Data(11,0.20,0.60,0.20)
#Compare_CFAR_Data()

#Algoritmo genético
#Parámetros:
#   cantidad de generaciones, porcentaje mutación, % cruces
#Métodos para cruce:
#   Cruzar el más apto con cada uno de los demás
#   El más apto con el más apto de lo menos aptos y así sucesivamente
#   Agarrar el porcentaje de cruce y empezar desde el centro para cruzar uno apto con uno no apto
#      y así hasta llegar al más apto con el menos apto (en el caso de 100% de cruce)












