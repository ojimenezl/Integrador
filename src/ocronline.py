## compute_input.py

import sys, json,requests, numpy as np
from sklearn.metrics import classification_report,confusion_matrix
#-- coding: iso-8859-1 --
import fitz
import re
from tabula import read_pdf
import tabula
import pandas as pd
import numpy as np
import csv
ngrup=2
def vali(y_ver,labels,ngrup):
 #################  METODOS DE VALIDACION ##################################
 from sklearn.metrics import classification_report,confusion_matrix
 matriz=confusion_matrix(y_ver,labels) #(y=verdaderas del data vs y_prediccion)-----------> IMPORTANTE ######
 #print(matriz)
 #matriz=np.delete(matriz,0,0)
 #matriz=np.delete(matriz,ngrup+1,1)
 

 #print("#######################################################################")
 #print(matriz,"\n")


 #print("---- Matriz Positive ----")
 P=np.diagonal(matriz)
 #print(P,"\n")

 #Suma filas
 #print("---- Matriz True Positive ----")
 TP=np.apply_along_axis(sum, 1, matriz)
 #print(TP,"\n")

 #Suma columnas
 #print("---- Matriz False Positive ----")
 FPC=np.apply_along_axis(sum, 0, matriz)
 #print(FPC,"\n")

 def FuncPrecision(P,FPC):
  precision=P/FPC
  return precision

 def FuncRecall(P,TP):
  recall=P/TP
  return recall

 def FuncFM():
  FM=2*((FuncPrecision(P,FPC)*FuncRecall(P,TP))/(FuncPrecision(P,FPC)+FuncRecall(P,TP)))
  return FM
 def FuncAccuracy(P,TP):
  PSuma=sum(P)
  TPSuma=sum(TP)
  accuracy=PSuma/TPSuma
  return accuracy


 A1=(FuncAccuracy(P,TP))
 


 P2=(FuncPrecision(P,FPC))
 P1=' , '.join(map(str, P2))

 R2=(FuncRecall(P,TP))
 R1=' , '.join(map(str, R2))

 F2=(FuncFM())
 F1=' , '.join(map(str, F2))

 return A1,P1,R1,F1



carp="public"
h= sys.argv[1]

pdf=h



import pandas as pd

df = pd.read_table('sentencias2.csv', 
                   sep=';', 
                   names=['label','sms_message'],encoding='iso-8859-1')


# Definir los documentos
documents = df
# Importar el contador de vectorizacion e inicializarlo
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
# Visualizar del objeto'count_vector' que es una instancia de 'CountVectorizer()'
#print(count_vector)

count_vector.fit(documents)
names = count_vector.get_feature_names()
#print(names)

doc_array = count_vector.transform(documents).toarray()
#print(doc_array)

frequency_matrix = pd.DataFrame(data=doc_array, columns=names)
#print(frequency_matrix)

# Dividir los datos en conjunto de entrenamiento y de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=1)



# Instantiate the CountVectorizer method
count_vector = CountVectorizer()
# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
tt=testing_data
predictions = naive_bayes.predict(testing_data)

############################################# NAIVE BAYES #################################################################
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
X_test1=[pdf]
#print(X_test1)

testing_data = count_vector.transform(X_test1)

#print("---testing_data = ",testing_data)

predictionsNB = naive_bayes.predict(testing_data)
#print("---prediccion = ",predictionsNB)


############################################# SVM #################################################################

from sklearn import svm

X = training_data
y = y_train
clf = svm.SVC()
clf.fit(X, y)
predictionsSVM=clf.predict(testing_data)
#print("---prediccion SVM-CLAS= ",predictionsSVM)


############################################ RL ###################
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0,solver='lbfgs',class_weight='balanced', max_iter=10000).fit(X, y)
predictionsRL=clf.predict(testing_data)


########################################### KNN ####################
from sklearn.neighbors import NearestCentroid
import numpy as np

clf = NearestCentroid()
clf.fit(X, y)
predictionsKNN=clf.predict(testing_data)





if(predictionsRL==1):
 res='GANA'
 #print(res)
else:
 res='PIERDE'
 #print(res) 


predictionsNB=''.join(map(str, predictionsNB))
predictionsSVM=''.join(map(str, predictionsSVM))
predictionsRL=''.join(map(str, predictionsRL))
predictionsKNN=''.join(map(str, predictionsKNN))
masdata={
"NB":predictionsNB,
"SVM":predictionsSVM,
"RL":predictionsRL,
"KNN":predictionsKNN,
"res":res
}




print(json.dumps(masdata))

sys.stdout.flush()
