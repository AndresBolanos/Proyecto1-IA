from sklearn.datasets import load_iris
import numpy as np

#Funcion para cargar los datos de Iris
def Load_Irirs_Data():    
    iris = load_iris()
    # Arreglos de numpy
    train_data = iris.data
    train_labels = iris.target

    return train_data

#Funcion para cargar los nombres de Iris
def Load_Iris_Names():
    iris = load_iris()
    train_names = iris.target_names
    return train_names

def Generate_W(filas, columnas):
    #Genero la matriz aleatoria con numeros entre 1 y 255
    #con la cantidad de filas y  columnas que recibe como parametro la funcion
    w = np.random.randint(1, 255, (filas, columnas))
    #retorna el w aleatorio
    return w 

# Funcion para multiplicar cada elemento de iris
# por el w generado aleatoriamente
def Compare_Iris_Data():
    X = Load_Irirs_Data() # Se trae todos los datos de iris
    Class = Load_Iris_Names() # Se trae todos los labels de iris
    W = Generate_W(len(X[0]), len(Class)+1) #Genera e w aleatorio
    numer_Items = len(X) #Largo de los datos
    Result_Array = [[0,0,0,0]] #No puedo inicializarlo vacio
    #Aplicamos la multiplicacion para cada uno de los elementos del data set
    for i in range(numer_Items):
        R = np.dot(W, X[i,:])
        Result_Array = np.append(Result_Array, [R], axis=0)
    Result_Array = np.delete(Result_Array, 0, 0) #Elimino la primera lista de ceros
    print(Result_Array)

Compare_Iris_Data()

