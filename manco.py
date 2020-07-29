import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from scipy import stats
import seaborn as sns; sns.set()
from sklearn import datasets #en caso de no usar estos datos usariamos pandas para importar datos de .cvs .sql .xlsx o de la red
#Lo que haremos ahora es crear una variable donde almacenar los datos
dataset=datasets.load_breast_cancer() #wueneo acá lo que tenemo es datasets es el comando donde estan almacenados los datos dentro de sklearn .load_ es para cargar el data y luego viene el nombre de la base de datos
print(dataset) #con print lo que hacemos es visualizar los datos
#listo ya con los datos vamos a ver que es lo que tenemos, necesitamos entender que hay alli para poder ver cual es el mejor modelo a implementar
print("información del dataset:")
print(dataset.keys()) #con .keys vemos que tiene los datos
print()
#entonces aca vemos que la base de datos tiene 8 comandos, una con datos, la segunda con las etiquetas, la tercera con frame, la cuarta con target_name que son los nombres de las etiquetas, la quinta con descripcion total del dataset, la sexta con feauture_names que nos dice todos los nombres de las colunas de los datos, y la ultima con filename
#ya con keys vamos a ver la descripcion de los datos
print("caracteristicas de los datos:")
print(dataset.DESCR) #Como podemos ver, esta base de datos tiene 569 datos, 30 atributos ó variables independientes 
x = dataset.data
#Para y utilizaremos los datos de las etiquetas
y= dataset.target
# 1 el tumor es maligno 0el tumor es benigno
## BUENO COMO SABEMOS QUE LOS DATOS ESTAN UNIDOS Y QUEREMOS SEPARARLO EN VARIABLES DE ENTRENAMIENTO Y EVALUACION IMPORTAREMOS LA LIBRERIA QUE NOS PERMITE HACER ESO###
from sklearn.model_selection import train_test_split
#ahora lo que haremos es separar los datos en entrenamiento (train) y en evaluacion(test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #el 0.2 es que le estamos diciendo al programa del conjunto de datos que le estoy dando el 20% sera para que me evalue el modelo
#como los datos no tienen las mismas magnitudes, lo mejor es que los escalemos, escalar es que los pongamos en las mismas condiciones 
#definimos el modelo a utilizar
from sklearn.svm import SVC
model = SVC(kernel ="linear", C=1E10)
#entrenamos el modelo
model.fit(x_train, y_train)
def plot_svc_decision_function(model, ax=None, plot_support=True):
    #Funcion de Vectores a 2D
    if ax is None:
        ax=plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()  
    x=np.linspace(xlim[0], xlim[1], 30)
    y=np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y,x)
    xy=np.vstack([X.ravel(), Y.ravel()]).T
    P= model.decision_function(xy).reshape(X.shape)
    ax.contour(X, Y, P, colors="k", levels=[-1,0,1], alpha=0.5, linestyle=["--","-","--"])
    if plot_support:
        ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=300, linewidth=1, facecolors="none")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim) 
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=50, cmap="icefire")
plot_svc_decision_function(model)
plt.show()
print(model.support_vectors_)
y_pred = model.predict(x_test)
print(y_pred)
print(y_test)
#verifico lo anterior con la matriz de confusion
from sklearn.metrics import confusion_matrix
matriz=confusion_matrix(y_test, y_pred)
print("matriz de confusion es:")
print(matriz) #wueno los resultados se interpretan así: en la diagonal principal, son los datos que se han predicho correctamente, en la diagonal secundaria los que presentaron errores, se deben sumar para saber el total
#otros métodos de precisión
from sklearn.metrics import precision_score
precision =precision_score(y_test, y_pred)
print("la precision del modelo es:")
print(precision)
#calculamos la exactitud del modelo
from sklearn.metrics import accuracy_score
exactitud=accuracy_score(y_test, y_pred)
print("la exactitud del modelo es:")
print(exactitud)
#calculamos la sensibilidad del modelo
from sklearn.metrics import recall_score
sensibilidad= recall_score(y_test, y_pred)
print("la sensibilidad del modelo es:")
print(sensibilidad)
#calculamos el F1 que es la combinacion de la sensibilidad y la exactitud
from sklearn.metrics import f1_score
puntajef1 = f1_score(y_test, y_pred)
print("Puntaje F1 del modelo:")
print(puntajef1)
#calculo la curva ROC - AUC del modelo
from sklearn.metrics import roc_auc_score
roc_auc=roc_auc_score(y_test, y_pred)
print("Curva ROC-AUC del modelo:")
print(roc_auc)
#Acá la metrica mas importante o relevante o bueno la que podemos usar es la curva ROC-AUC ó el F1
