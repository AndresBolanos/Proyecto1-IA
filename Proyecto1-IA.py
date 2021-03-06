# -*- coding: cp1252 -*-

from sklearn.datasets import load_iris
import numpy as np
import pickle
import collections
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    # mu, sigma = 4,2 
    mu, sigma = 4,2 # mean and standard deviation  
    W = []
    for i in range (filas):
        W.append(np.absolute(np.random.normal(mu, sigma, columnas).tolist()))   
    W = np.array(W)
    #print(W)
    return W

#Funcion para generar el W aleatorio en CFAR-10 y Iris con una distribucion normal
def Generate_W_CFAR(filas, columnas):
    #Genero la matriz aleatoria con numeros entre 1 y 255
    #con la cantidad de filas y  columnas que recibe como parametro la funcion
    #w = np.random.randint(1, 255, (filas, columnas)) #Random con decimales #####CIFAR###
    # mu, sigma = 5,2
    mu, sigma = 50,25 # mean and standard deviation  
    W = []
    for i in range (filas):
        W.append(np.absolute(np.random.normal(mu, sigma, columnas).tolist()))   
    W = np.array(W)
    #print(W)
    return W

#################################### MIX W #######################################
#Funcion para cruzar en el set de datos de Iris
#Cruza el m�s apto con el m�s apto de los menos aptos
def Cruzar_Generacion(Lista_W, Lista_Loss, Lista_Indices, X, Index):
    W_Loss_Indices = []                                       #Lista que tiene lista de W's, otra de los Loss y la otra de los indices
    #Si la cantidad de W no es par, el m�s �ptimo pasa sin cruzarse
    
    if len(Lista_Indices)%2 != 0:
        W_Loss_Indices.append(Lista_W[Lista_Indices[0]])
        Lista_Indices = Lista_Indices[1:]                             #Quita el m�s �ptimo para no cruzarlo
    #Solo va a recorrer la mitad m�s optima y compararlos con la otra mitad menos �ptima
    for i in range(len(Lista_Indices)/2):
        nuevoW1 = []
        nuevoW2 = []
        PosW_NoOptimo = i+(len(Lista_Indices)/2)                      #En lista de len=10 se le suma la mitad y ser�a: 0 con 5, 1 con 6, 2 con 7...
        W_Optimo = Lista_W[Lista_Indices[i]]                          #Entre m�s bajo el i, m�s �ptimo el W
        W_NoOptimo = Lista_W[Lista_Indices[PosW_NoOptimo]]
        
        #La clase que mejor reconozca la va a dejar igual en un nuevoW, en el otro va una combinaci�n
        for j in range(len(W_Optimo)):                                                          #Recorre la cantidad de clases que tiene cada W (3 en el caso de IRIS)
            NuevaClase1 = []                                                                    #El nuevo W con los genes m�s �ptimos
            NuevaClase2 = []
            if (Lista_Loss[Lista_Indices[i]][j]<=Lista_Loss[Lista_Indices[PosW_NoOptimo]][j]):  #Compara los Loss de las clases de los W y agrega el mejor
                NuevaClase1.append((W_Optimo[j]).tolist())          
            else:
                NuevaClase1.append((W_NoOptimo[j]).tolist())
                
            regula = 1                                                                           #1 para agarrar dato de la clase 1 y -1 de la clase 2
            for k in range(len(W_Optimo[j])):                                                    #Recorre cada dato referente a cada clase (p�talos, hojas... en IRIS)
                if (regula == 1):                                  
                    NuevaClase2.append(W_Optimo[j][k])                                           #Agrega un gen del W �ptimo al nuevo W
                else:
                    NuevaClase2.append(W_NoOptimo[j][k])                                         #Agrega un gen del W no �ptimo al nuevo W
                regula *= -1

            nuevoW1.append(NuevaClase1[0])                                                       #El nuevo W con las mejores clases
            nuevoW2.append(NuevaClase2)                                                          #El nuevo W con el cruce de genes de los padres
        
        W_Loss_Indices.append(np.array(nuevoW1))                                                 #Se agrega el W con las clases m�s optimas
        W_Loss_Indices.append(np.array(nuevoW2))                                                 #Se agrega el W cruzado

    return W_Loss_Indices



def Cruzar_Generacion2(Lista_W, Lista_Loss, Lista_Indices, X, Index):
    W_Loss_Indices = []                                       #Lista que tiene lista de W's, otra de los Loss y la otra de los indices
    #Si la cantidad de W no es par, el m�s �ptimo pasa sin cruzarse
    
    if len(Lista_Indices)%2 != 0:
        W_Loss_Indices.append(Lista_W[Lista_Indices[0]])
        Lista_Indices = Lista_Indices[1:]                             #Quita el m�s �ptimo para no cruzarlo
    #Solo va a recorrer la mitad menos optima y compararlos con la otra mitad m�s �ptima
    for i in range(len(Lista_Indices)/2):
        nuevoW1 = []
        nuevoW2 = []
        PosW_NoOptimo = len(Lista_Indices)-1-i                          #En lista de len=10 se le suma la mitad y ser�a: 0 con 5, 1 con 6, 2 con 7...
        PosW_Optimo = len(Lista_Indices)-1 -i-(len(Lista_Indices)/2) 
        W_Optimo = Lista_W[Lista_Indices[PosW_Optimo]]                          #Entre m�s bajo el i, m�s �ptimo el W
        W_NoOptimo = Lista_W[Lista_Indices[PosW_NoOptimo]]
        
        #La clase que mejor reconozca la va a dejar igual en un nuevoW, en el otro va una combinaci�n
        for j in range(len(W_Optimo)):                                                          #Recorre la cantidad de clases que tiene cada W (3 en el caso de IRIS)
            NuevaClase1 = []                                                                    #El nuevo W con los genes m�s �ptimos
            NuevaClase2 = []
            if (Lista_Loss[Lista_Indices[i]][j]<=Lista_Loss[Lista_Indices[PosW_NoOptimo]][j]):  #Compara los Loss de las clases de los W y agrega el mejor
                NuevaClase1.append((W_Optimo[j]).tolist())          
            else:
                NuevaClase1.append((W_NoOptimo[j]).tolist())
                
            regula = 1                                                                           #1 para agarrar dato de la clase 1 y -1 de la clase 2
            for k in range(len(W_Optimo[j])):                                                    #Recorre cada dato referente a cada clase (p�talos, hojas... en IRIS)
                if (regula == 1):                                  
                    NuevaClase2.append(W_Optimo[j][k])                                           #Agrega un gen del W �ptimo al nuevo W
                else:
                    NuevaClase2.append(W_NoOptimo[j][k])                                         #Agrega un gen del W no �ptimo al nuevo W
                regula *= -1

            nuevoW1.append(NuevaClase1[0])                                                       #El nuevo W con las mejores clases
            nuevoW2.append(NuevaClase2)                                                          #El nuevo W con el cruce de genes de los padres
        
        W_Loss_Indices.append(np.array(nuevoW1))                                                 #Se agrega el W con las clases m�s optimas
        W_Loss_Indices.append(np.array(nuevoW2))                                                 #Se agrega el W cruzado

    return W_Loss_Indices


def Cruzar_Generacion3(Lista_W, Lista_Loss, Lista_Indices, X, Index):
    W_Loss_Indices = []                                       #Lista que tiene lista de W's, otra de los Loss y la otra de los indices
    #Si la cantidad de W no es par, el m�s �ptimo pasa sin cruzarse
    
    if len(Lista_Indices)%2 != 0:
        W_Loss_Indices.append(Lista_W[Lista_Indices[0]])
        Lista_Indices = Lista_Indices[1:]                             #Quita el m�s �ptimo para no cruzarlo
    #Solo va a recorrer la mitad menos optima y compararlos con la otra mitad m�s �ptima
    contador = 0  #Para saber la posici�n de los padres
    for i in range(len(Lista_Indices)/2):
        nuevoW1 = []
        nuevoW2 = []
        PosW_NoOptimo = i+contador+1                                         #En lista de len=10 se le suma la mitad y ser�a: 0 con 5, 1 con 6, 2 con 7...
        PosW_Optimo = i+contador 
        W_Optimo = Lista_W[Lista_Indices[PosW_Optimo]]                #Entre m�s bajo el i, m�s �ptimo el W
        W_NoOptimo = Lista_W[Lista_Indices[PosW_NoOptimo]]

        contador += 1
        
        #La clase que mejor reconozca la va a dejar igual en un nuevoW, en el otro va una combinaci�n
        for j in range(len(W_Optimo)):                                                          #Recorre la cantidad de clases que tiene cada W (3 en el caso de IRIS)
            NuevaClase1 = []                                                                    #El nuevo W con los genes m�s �ptimos
            NuevaClase2 = []
            if (Lista_Loss[Lista_Indices[i]][j]<=Lista_Loss[Lista_Indices[PosW_NoOptimo]][j]):  #Compara los Loss de las clases de los W y agrega el mejor
                NuevaClase1.append((W_Optimo[j]).tolist())          
            else:
                NuevaClase1.append((W_NoOptimo[j]).tolist())
                
            regula = 1                                                                           #1 para agarrar dato de la clase 1 y -1 de la clase 2
            for k in range(len(W_Optimo[j])):                                                    #Recorre cada dato referente a cada clase (p�talos, hojas... en IRIS)
                if (regula == 1):                                  
                    NuevaClase2.append(W_Optimo[j][k])                                           #Agrega un gen del W �ptimo al nuevo W
                else:
                    NuevaClase2.append(W_NoOptimo[j][k])                                         #Agrega un gen del W no �ptimo al nuevo W
                regula *= -1

            nuevoW1.append(NuevaClase1[0])                                                       #El nuevo W con las mejores clases
            nuevoW2.append(NuevaClase2)                                                          #El nuevo W con el cruce de genes de los padres
        
        W_Loss_Indices.append(np.array(nuevoW1))                                                 #Se agrega el W con las clases m�s optimas
        W_Loss_Indices.append(np.array(nuevoW2))                                                 #Se agrega el W cruzado

    return W_Loss_Indices


def Cruzar_Generacion4(Lista_W, Lista_Loss, Lista_Indices, X, Index):
    W_Loss_Indices = []                                       #Lista que tiene lista de W's, otra de los Loss y la otra de los indices
    #Si la cantidad de W no es par, el m�s �ptimo pasa sin cruzarse
    
    if len(Lista_Indices)%2 != 0:
        W_Loss_Indices.append(Lista_W[Lista_Indices[0]])
        Lista_Indices = Lista_Indices[1:]                             #Quita el m�s �ptimo para no cruzarlo
    #Solo va a recorrer la mitad menos optima y compararlos con la otra mitad m�s �ptima
    contador = 0  #Para saber la posici�n de los padres
    for i in range(len(Lista_Indices)/2):
        nuevoW1 = []
        nuevoW2 = []
        PosW_NoOptimo = len(Lista_Indices)-1-i                                       #En lista de len=10 se le suma la mitad y ser�a: 0 con 5, 1 con 6, 2 con 7...
        PosW_Optimo = len(Lista_Indices)-1-i-contador-1
        W_Optimo = Lista_W[Lista_Indices[PosW_Optimo]]                #Entre m�s bajo el i, m�s �ptimo el W
        W_NoOptimo = Lista_W[Lista_Indices[PosW_NoOptimo]]

        contador += 1
        
        #La clase que mejor reconozca la va a dejar igual en un nuevoW, en el otro va una combinaci�n
        for j in range(len(W_Optimo)):                                                          #Recorre la cantidad de clases que tiene cada W (3 en el caso de IRIS)
            NuevaClase1 = []                                                                    #El nuevo W con los genes m�s �ptimos
            NuevaClase2 = []
            if (Lista_Loss[Lista_Indices[i]][j]<=Lista_Loss[Lista_Indices[PosW_NoOptimo]][j]):  #Compara los Loss de las clases de los W y agrega el mejor
                NuevaClase1.append((W_Optimo[j]).tolist())          
            else:
                NuevaClase1.append((W_NoOptimo[j]).tolist())
                
            regula = 1                                                                           #1 para agarrar dato de la clase 1 y -1 de la clase 2
            for k in range(len(W_Optimo[j])):                                                    #Recorre cada dato referente a cada clase (p�talos, hojas... en IRIS)
                if (regula == 1):                                  
                    NuevaClase2.append(W_Optimo[j][k])                                           #Agrega un gen del W �ptimo al nuevo W
                else:
                    NuevaClase2.append(W_NoOptimo[j][k])                                         #Agrega un gen del W no �ptimo al nuevo W
                regula *= -1

            nuevoW1.append(NuevaClase1[0])                                                       #El nuevo W con las mejores clases
            nuevoW2.append(NuevaClase2)                                                          #El nuevo W con el cruce de genes de los padres
        
        W_Loss_Indices.append(np.array(nuevoW1))                                                 #Se agrega el W con las clases m�s optimas
        W_Loss_Indices.append(np.array(nuevoW2))                                                 #Se agrega el W cruzado

    return W_Loss_Indices
                    

        
    
#################################### MUTATION W #######################################

#Funcion de mutacion 1
#Selecciona con base al porcentaje de mutacion una determinada cantidad de W
#Genera un nuevo W con valores random de posiciones random segun la cantidad q se indica en el procentaje de mutacion 2
#Ese nuevo W se cambia por el que tenia el Loss mas bajo
def Mutacion_1(Lista_W,Lista_Indices,Lista_Loss,mutacion_1, mutacion_2):
    cantidad_W = int(round(len(Lista_W) * mutacion_1)/2) #Calcula la cantidad de W a escoger segun el procentaje
    cantidad_Genes = int(round(len(Lista_W[0][0])) * mutacion_2) #Indica la cantidad de genes que se van a mutar dentro de la W
    posicion_1 = ((len(Lista_Indices)/2) -1)+len(Lista_Indices)%2 #Seleccion los masomenos aptos para muta
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
    cantidad_W = int(round(len(Lista_W) * mutacion_1)/2) #Calcula la cantidad de W a escoger segun el procentaje
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
Efectividad = 0
#Funcion para calcular el loss de cada clase y el total de W
#etorna un arreglo con el loss de cada clase y en la ultima posicion el loss de W
def Calculo_Loss(W,X,Labels):
    cont = len(collections.Counter(Labels).items())                 #Contar la cantidad de clases que hay
    Lista_Loss = np.zeros((cont+1), dtype=long)
    L = 0
    global Efectividad                                          #Genera un arreglo de ceros del largos de las clases
    Efectividad = 0
    for i in range(len(X)):
        R = np.dot(W, X[i])                                         #Multiplicacion de W por cada imagen
        if (np.argmax(R) == Labels[i]):
            Efectividad +=1
        loss = Hinge_Loss(R,Labels[i])
        L = L + loss                                                #Loss para el vector solucion
        Lista_Loss[Labels[i]] = Lista_Loss[Labels[i]]+ loss         #Guarda el loss en la respectiva posicion  
    Lista_Loss[cont] = L/len(X)
    Efectividad = float(Efectividad)
    Efectividad = Efectividad/len(X)                                             #Hace la sumatoria de todos los loss y lo pone en la ultima posicion
    return Lista_Loss

'''def Comprobar_Optimo(valOptimo, Lista_Loss):
    for i in range(len(Lista_Loss)):
        #print "W[",i,"]: ", Lista_Loss[i][-1]
        if Lista_Loss[i][-1] <= valOptimo:
            return i
    return -1                                                  #si no encontr� ning�n �ptimo
'''
#mutacion_1: cantidad de Wi que se van a mutar
#mutacion_2: cantidad de cambio en los genes de cada Wi
def Compare_Iris_Data(k,mutacion_1,mutacion_2,cruce):
    #Variable del grafico
    global Efectividad
    Efectividad_aux = 0
    eje_X = []
    eje_Y = []
    eje_X1 = []
    eje_Y1 = []
    valOptimo = 0.5                       #Porcentaje �ptimo
    N = 100                                 #Cantidad de generaciones a evaluar 
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
    cont = 1
    for i in range(N):
        print "\nGeneraci�n: ", i+1            #Comprueba si ya hay alg�n W �ptimo
        print "Optimo: ", valOptimo
        if Efectividad_aux >= valOptimo and i > 1:
            plt.title('Generation VS Efectividad')
            plt.subplot(2,1,1)   
            plt.plot(eje_X1,eje_Y1)
            plt.ylabel('% Efectividad')
            plt.grid(True)
            plt.subplot(2,1,2)   
            plt.plot(eje_X,eje_Y)
            plt.xlabel('Generation')
            plt.ylabel('Loss')
            plt.grid(True)  # Activa cuadr�cula del gr�fico pero no se muestra
            plt.show()
            return 0
        else:
            Lista_W = Cruzar_Generacion3(Lista_W, Lista_Loss, Lista_Indices, X, Index)
            
        Lista_Indices = []
        for i in range(k):
            Lista_Loss[i] = Calculo_Loss(Lista_W[i],X,Index)
            Lista_Indices = Insertar_Indices(Lista_Indices,i,Lista_Loss)
            
        Lista_W = Mutacion_2(Lista_W,Lista_Indices,Lista_Loss,mutacion_1, mutacion_2)  #Mutaci�n de la generaci�n
        
        Lista_Indices = []
        for i in range(k):
            Lista_Loss[i] = Calculo_Loss(Lista_W[i],X,Index)
            Lista_Indices = Insertar_Indices(Lista_Indices,i,Lista_Loss)
            if Efectividad_aux < Efectividad:
                Efectividad_aux = Efectividad 
        

        print "Efectividad: ", Efectividad_aux
          
        #Este codigo es si se quiere sacar el promedio general del loss de toda la poblacion
        
        """promedio_Loss = 0
        for i in range(len(Lista_Loss)):
            promedio_Loss += Lista_Loss[i][len(Lista_Loss[i])-1]
        print("Promedio")
        promedio_Loss = promedio_Loss/len(Lista_Loss)
        print(promedio_Loss)"""
        eje_X = eje_X+[cont]
        eje_X1 = eje_X1+[cont]
        eje_Y.append(Lista_Loss[Lista_Indices[0]][-1])
        eje_Y1.append(Efectividad_aux)
        cont+=1
        
    plt.subplot(2,1,1)   
    plt.plot(eje_X1,eje_Y1)
    plt.ylabel('% Efectividad')
    plt.title('Generation VS Efectividad')
    plt.grid(True)
    plt.subplot(2,1,2)   
    plt.plot(eje_X,eje_Y)
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.grid(True)  # Activa cuadr�cula del gr�fico pero no se muestra
    plt.show()
    print "\nNo hay �ptimo"
        
    return 0

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

#Funcion para generar cada imagen para mostrar                
def GenerarImagenes(img1,img2,img3,img4):
    cont = 0
    matriz1 = []
    matriz2 = []
    matriz3 = []
    matriz4 = []
    img1_aux = []
    img2_aux = []
    img3_aux = []
    img4_aux = []
    for i in range(1024):
        if cont == 31:
            matriz1.append(img1_aux)
            matriz2.append(img2_aux)
            matriz3.append(img3_aux)
            matriz4.append(img4_aux)
            cont = 0
            img1_aux = []
            img2_aux = []
            img3_aux = []
            img4_aux = []
        else:
            img1_aux.append(img1[i])
            img2_aux.append(img2[i])
            img3_aux.append(img3[i])
            img4_aux.append(img4[i])
            cont+=1
    return matriz1,matriz2,matriz3,matriz4
    
# Funcion para multiplicar cada elemento de cfar
# por el w generado aleatoriamente
def Compare_CFAR_Data(k,mutacion_1,mutacion_2,cruce):
    global Efectividad
    Efectividad_aux = 0
    # Se trae todos los datos de cfar-10
    #Labels_Test = Load_CFAR_Labels_Test()
    #X_Test = Load_CFAR_Data_Test(Labels_Test)
    
    #Labels_Test = get_Labels(Labels_Test)
    eje_X = []
    eje_Y = []
    eje_X1 = []
    eje_Y1 = []
    valOptimo = 0.35                                            #% �ptimo
    N = 15                                                    #Cantidad de generaciones a evaluar 
    Lista_Loss = []                                             #Guarda el loss para cada clase
    L = 0;                                                      #Contiene la sumatoria de aplicar hinge-loss a cada elemento
    Index = Load_CFAR_Labels()
    X = Load_CFAR_Data(Index)
    Index = get_Labels(Index)
    #Se trae todos los indices de resultado de iris
    Class = len(collections.Counter(Index).items())               #Se trae todos los nombres de iris
    number_Items = len(X)                                         #Largo de los datos
    Lista_W = []
    Lista_Loss = []
    Lista_Indices = []
    for i in range(k):
        Lista_W.append(Generate_W_CFAR(Class,len(X[0])))                 #Genera el w aleatorio ##Automaticamente en la funcion que genera el +1 del Bias Trick
        Lista_Loss.append(Calculo_Loss(Lista_W[i],X,Index))          #Guarda la lista de loss de clase y de W para cada W generado
        Lista_Indices = Insertar_Indices(Lista_Indices,i,Lista_Loss) #Guarda los �ndices ordenados de los W del loss menor al mayor

    #print(Lista_Loss)
    #print(Lista_Indices)
    #Algoritmo gen�tico con N repeticiones o hasta que encuentre uno �ptimo con Loss <= Optimo
    cont = 1
    for i in range(N):
        print "\nGeneraci�n: ", i+1             #Comprueba si ya hay alg�n W �ptimo
        if Efectividad_aux>valOptimo and i > 1:
            plt.title('Generation VS Loss')
            plt.subplot(2,1,1)   
            plt.plot(eje_X1,eje_Y1)
            plt.ylabel('% Efectividad')
            plt.grid(True)
            plt.subplot(2,1,2)   
            plt.plot(eje_X,eje_Y)
            plt.xlabel('Generation')
            plt.ylabel('Loss')
            plt.grid(True)  # Activa cuadr�cula del gr�fico pero no se muestra
            plt.show()
            return 0
        else:
            Lista_W = Cruzar_Generacion3(Lista_W, Lista_Loss, Lista_Indices, X, Index)
            
        Lista_Indices = []
        for i in range(k):
            Lista_Loss[i] = Calculo_Loss(Lista_W[i],X,Index)
            Lista_Indices = Insertar_Indices(Lista_Indices,i,Lista_Loss)
            
        Lista_W = Mutacion_2(Lista_W,Lista_Indices,Lista_Loss,mutacion_1, mutacion_2)  #Mutaci�n de la generaci�n
        
        Lista_Indices = []
        for i in range(k):
            Lista_Loss[i] = Calculo_Loss(Lista_W[i],X,Index)
            Lista_Indices = Insertar_Indices(Lista_Indices,i,Lista_Loss)

            if Efectividad_aux < Efectividad:
                Efectividad_aux = Efectividad 
        

        print "Efectividad: ", Efectividad_aux
          
        #Este codigo es si se quiere sacar el promedio general del loss de toda la poblacion
        """
        promedio_Loss = 0
        for i in range(len(Lista_Loss)):
            promedio_Loss += Lista_Loss[i][len(Lista_Loss[i])-1]
        print("Promedio")
        promedio_Loss = promedio_Loss/len(Lista_Loss)
        print(promedio_Loss)"""
        
        #Este codigo es para guardar el loss del W mas apto
        eje_X = eje_X+[cont]
        eje_X1 = eje_X1+[cont]
        eje_Y.append(Lista_Loss[Lista_Indices[0]][-1])
        eje_Y1.append(Efectividad_aux)
        cont+=1
        
    pos = Lista_Indices[0]
    res = GenerarImagenes(Lista_W[pos][0],Lista_W[pos][1],Lista_W[pos][2],Lista_W[pos][3])
    fig=plt.figure()
    for i in range(4):
        fig.add_subplot(2, 2, i+1)
        plt.imshow(res[i], cmap= plt.cm.binary)
    plt.show()
    
    plt.subplot(2,1,1)   
    plt.plot(eje_X1,eje_Y1)
    plt.ylabel('% Efectividad')
    plt.title('Generation VS Efectividad')
    plt.grid(True)
    plt.subplot(2,1,2)   
    plt.plot(eje_X,eje_Y)
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.grid(True)  # Activa cuadr�cula del gr�fico pero no se muestra
    plt.show()
    print "\nNo hay �ptimo"
         
    return 0
             
    
#################################### HINGE LOSS FUNCTION #######################################    
#Recibe el vector s y la posicion del yi de la clase optimo del arreglo s
def Hinge_Loss(s,yi):
    hinge_loss = 0;
    for i in range(len(s)):
        if (i != yi):
            hinge_loss += np.sum(np.maximum(0,s[i]-s[yi]+1),axis=0)
    return hinge_loss

#Compare_Iris_Data(50,0.8,0.6,0.8)   #(Tama�oPoblaci�n,CantWParaMutar,Cu�ntoCambioEnGenes,cruce)
Compare_CFAR_Data(20,0.8,0.60,0.8)

#Algoritmo gen�tico
#Par�metros:
#   cantidad de generaciones, porcentaje mutaci�n, % cruces
#M�todos para cruce:
#   Cruzar el m�s apto con cada uno de los dem�s
#   El m�s apto con el m�s apto de lo menos aptos y as� sucesivamente
#   Agarrar el porcentaje de cruce y empezar desde el centro para cruzar uno apto con uno no apto
#      y as� hasta llegar al m�s apto con el menos apto (en el caso de 100% de cruce)












