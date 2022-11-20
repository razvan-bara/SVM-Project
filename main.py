import numpy as np
import pandas
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score


#Citirea datelor de intrare si datelor de testare.Aranjarea acestora intr-un array
data_in=pandas.read_csv("adultdatefin.csv")
date_de_test=pandas.read_csv("adulttest.csv")
data_in=np.asarray(data_in)
date_de_test=np.asarray(date_de_test)

#Eliminarea esantioanelor cu atribute care lipsesc pentru ambele seturi de date

for i in range(len(data_in)):
    if i<len(data_in):
        for j in range(len(data_in[i])):
            if data_in[i][j]==" ?":
                data_in=np.delete(data_in,i,0)
                
for i in range(len(date_de_test)):
    if i<len(date_de_test):
        for j in range(len(date_de_test[i])):
            if date_de_test[i][j]==" ?":
                date_de_test=np.delete(date_de_test,i,0)

#Preprocesarea atributelor: transformarea lor in numere si normalizarea
#acestor numere

le_a = preprocessing.LabelEncoder()
for i in range(14):
    data_in[:,i] = le_a.fit_transform(data_in[:,i])
    data_in[:,i]=preprocessing.normalize([data_in[:,i]])
    
le_t = preprocessing.LabelEncoder()
for i in range(14):
    date_de_test[:,i] = le_t.fit_transform(date_de_test[:,i])
    date_de_test[:,i]=preprocessing.normalize([date_de_test[:,i]])

#Separarea atributelor de etichete
#datele de antrenare
date_a=data_in[:,:14]
etichete_a=data_in[:,14]
#datele de testare
date_t=date_de_test[:,:14]
etichete_t=date_de_test[:,14]



#Specificarea tipului de date al etichetelor deoarece SVM-ul nu poate identifica tipul
etichete_a=etichete_a.astype('int')
etichete_t=etichete_t.astype('int')

#Crearea unui array cu valorile posibile ale costului,
#pentru a facilita varierea acestuia
costs_array = np.array([0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128])

#Antrenarea SVC-ului, prezicerea etichetelor datelor destinate testarii
#si obtinerea acuratetii penru fiecare valoare a costului
for i in range(13):
    clf=svm.SVC(kernel='linear',C=costs_array[i])
    clf.fit(date_a,etichete_a)
    predictii=clf.predict(date_t)
    print("Pentru costul egal cu "+str(costs_array[i])+" obtinem acuratetea:")
    print("{0:.3f}".format(accuracy_score(etichete_t,predictii)))
