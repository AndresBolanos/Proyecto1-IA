# -*- coding: cp1252 -*-

from sklearn.datasets import load_iris
import numpy as np
import pickle
import collections

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
#Cruza el m�s apto con el m�s apto de los menos aptos
def Cruzar_Generacion(Lista_W, Lista_Loss, Lista_Indices):
    nuevoW = []
    #Si la cantidad de W no es par, el m�s �ptimo pasa sin cruzarse
    if len(Lista_Indices)%2 != 0:
        nuevoW.append(Lista_W[0])
        Lista_Indices = Lista_Indices[1:]                #Quita el m�s �ptimo para no cruzarlo
    #Solo va a recorrer la mitad m�s optima y compararlos con la otra mitad menos �ptima
    for i in range(len(Lista_Indices)/2): 
        W_Optimo = Lista_W[i]                            #Entre m�s bajo el i, m�s �ptimo el W
        W_NoOptimo = Lista_W[i+(len(Lista_Indices)/2)]   #En lista de len=10 se le suma la mitad y ser�a: 0 con 5, 1 con 6, 2 con 7...

        #Ser�a ingresar a cada elemento de esas 2 matrices y cruzarlos como hab�amos
        #hecho en la hoja del cuaderno
        
        
            
        
    
#################################### MUTATION W #######################################
#Funcion para mutar una fila del set de iris
#Muta la fila con mas Loss



#################################### COMPARISON DATA FUNCTIONS #######################################    

#Funcion para calcular el loss de cada clase y el total de W
#etorna un arreglo con el loss de cada clase y en la ultima posicion el loss de W
def Calculo_Loss(W,X,Labels): 
    cont = len(collections.Counter(Labels).items())            #Contar la cantidad de clases que hay
    Lista_Loss = np.zeros((cont+1), dtype=int)                 #Genera un arreglo de ceros del largos de las clases
    for i in range(len(X)):
        R = np.dot(W, X[i])                                    #Multiplicacion de W por cada imagen
        L = Hinge_Loss(R,Labels[i])                            #Loss para el vector solucion
        Lista_Loss[Labels[i]] = Lista_Loss[Labels[i]]+ L       #Guarda el loss en la respectiva posicion  
    Lista_Loss[cont] = np.sum(Lista_Loss)                      #Hace la sumatoria de todos los loss y lo pone en la ultima posicion
    return Lista_Loss

def Comprobar_Optimo(valOptimo, Lista_Loss):
    for i in range(len(Lista_Loss)-1):
        if Lista_Loss[i][len(Lista_Loss[0])-1] <= valOptimo:
            return i
    return -1                                                  #si no encontr� ning�n �ptimo

def Compare_Iris_Data(k,mutacion,cruce):
    valOptimo = 100                         #Loss �ptimo
    N = 1000                                #Cantidad de generaciones a evaluar 
    Lista_Loss = []                         #Guarda el loss para cada clase
    L = 0;                                  #Contiene la sumatoria de aplicar hinge-loss a cada elemento
    X = Load_Irirs_Data()                   #Se trae todos los datos de iris
    Index = Load_Iris_Index()               #Se trae todos los indices de resultado de iris
    Class = Load_Iris_Names()               #Se trae todos los nombres de iris
    number_Items = len(X)                   #Largo de los datos
    Lista_W = []
    Lista_Loss = []
    Lista_Indices = []
    for i in range(k):
        Lista_W.append(Generate_W(len(Class),len(X[0])))             #Genera el w aleatorio ##Automaticamente en la funcion que genera el +1 del Bias Trick
        Lista_Loss.append(Calculo_Loss(Lista_W[i],X,Index))          #Guarda la lista de loss de clase y de W para cada W generado
        Lista_Indices = Insertar_Indices(Lista_Indices,i,Lista_Loss) #Guarda los �ndices ordenados de los W del loss menor al mayor
        
    #Algoritmo gen�tico con N repeticiones o hasta que encuentre uno �ptimo con Loss <= Optimo
    for i in range(N):
        optimo = Comprobar_Optimo(valOptimo, Lista_Loss)             #Comprueba si ya hay alg�n W �ptimo
        if optimo != -1:
            return ("El optimo es: ", Lista_W[optimo])
        else:
            Lista_W = Cruzar_Generacion(Lista_W, Lista_Loss, Lista_Indices)
    return ("No hay �ptimo")
    print(Lista_Loss)

#Acomoda el indice en la lista de �ndices dependiendo del Loss. Del menor Loss al mayor
def Insertar_Indices(Lista_Indices,i,Lista_Loss):
    #Si la lista est� vac�a no compara, solo lo coloca
    if Lista_Indices == []: 
        Lista_Indices.append(i)
        return Lista_Indices
    else:
        #Compara hasta enontrar un Loss mayor en la lista y colocar el indice antes de ese
        for j in range(len(Lista_Indices)): 
            if Lista_Loss[Lista_Indices[j]][len(Lista_Loss[0])-1] > Lista_Loss[i][len(Lista_Loss[0])-1]:
                Lista_Indices = Lista_Indices[:j]+[i]+Lista_Indices[j:]
                return Lista_Indices
            #Si no encontr� un Loss mayor lo coloca al final de la lista
            else:
                if j == len(Lista_Indices)-1:
                    Lista_Indices = Lista_Indices+[i]
                    return Lista_Indices

                
    
# Funcion para multiplicar cada elemento de cfar
# por el w generado aleatoriamente
def Compare_CFAR_Data():
    Labels = Load_CFAR_Labels()
    X = Load_CFAR_Data(Labels);                         # Se trae todos los datos de cfar-10
    Labels_Test = Load_CFAR_Labels_Test()
    X_Test = Load_CFAR_Data_Test(Labels_Test)
    Labels = get_Labels(Labels)
    Labels_Test = get_Labels(Labels_Test)
    print("Entrenamiento largo vector "+str(len(X[0])))
    print("Labels  "+str(len(Labels)))
    print("Prueba largo vector "+str(len(X_Test[0])))
    print("Labels  "+str(len(Labels_Test)))
    #Pongo 4 porque son solo 4 clases hay q mapearlo con el dato bien
    W = Generate_W(4,len(X[0]))                         #Genera el w aleatorio ##Automaticamente en la funcion que genera el +1 del Bias Trick
    number_Items = len(X)                               #Largo de los datos
    L = 0;                                              #Contiene la sumatoria de aplicar hinge-loss a cada elemento
    for i in range(1000):                               #El valor 1000 es la cantidad de iteraciones q permite hacer. Es un hiperparametro
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

Compare_Iris_Data(4,20,20)
#Compare_CFAR_Data()

#Algoritmo gen�tico
#Par�metros:
#   cantidad de generaciones, porcentaje mutaci�n, % cruces
#M�todos para cruce:
#   Cruzar el m�s apto con cada uno de los dem�s
#   El m�s apto con el m�s apto de lo menos aptos y as� sucesivamente
#   Agarrar el porcentaje de cruce y empezar desde el centro para cruzar uno apto con uno no apto
#      y as� hasta llegar al m�s apto con el menos apto (en el caso de 100% de cruce)












