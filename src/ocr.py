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
 matriz=confusion_matrix(y_ver,labels) 


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

########################################### EXTRACCION PDF #################################3

carp="public"
pdf= sys.argv[1]
pdf=carp+pdf
df= read_pdf(pdf, pages="2")
tabula.convert_into(pdf,("frmXYelim-291272" + '.csv'),output_format="csv",pages="2")
pdf_documento=pdf
documento= fitz.open(pdf_documento)
pagina = documento.loadPage(1)
text= pagina.getText("text")
doc=(re.sub('[!?@#$()-.,;:*/0-9%"]+',' ',text.lower())).split()


################################### WHILE PARA OBTNER EL TEXTO DESEADO ############
c=0
m=0
hecho=[]
demanda=""

while (c<len(doc)): 
  if( doc[c] == "hecho"):
   c=c+1
   while (c<len(doc)):
    if( doc[c] == "sumario"):
     c=len(doc)
    else: 
     hecho.append(doc[c])
     demanda=demanda+doc[c]+" "
     c=c+1
  c=c+1




pagina = documento.loadPage(0)
text= pagina.getText("text")
doc=(re.sub('[!?#$(),;:"]+',' ',text.lower())).split()
#print(docS)
#doc=[]
#for m in docS:
# doc.append(normalize(m))

c=0
m=0
hecho=[]
nombrecedula=""

while (c<len(doc)): 
  if( doc[c] == "pasaporte"):
   c=c+1
   while (c<len(doc)):
    if( doc[c] == "edad"):
     c=len(doc)
    else: 
     hecho.append(doc[c])
     nombrecedula=nombrecedula+doc[c]+" "
     c=c+1
  c=c+1
#print(nombrecedula)


c=0
m=0
hecho=[]
celular=""

while (c<len(doc)): 
  if( doc[c] == "celular"):
   c=c+1
   while (c<len(doc)):
    if( doc[c] == "correo"):
     c=len(doc)
    else: 
     hecho.append(doc[c])
     celular=celular+doc[c]+" "
     c=c+1
  c=c+1

#print(celular)

c=0
m=0
hecho=[]
correo=""

while (c<len(doc)): 
  if( doc[c] == "judicial"):
   c=c+1
   while (c<len(doc)):
    if( doc[c] == "1.2"):
     c=len(doc)
    else: 
     hecho.append(doc[c])
     correo=correo+doc[c]+" "
     c=c+1
  c=c+1

#print(correo)

c=0
m=0
hecho=[]
nombrecedula2=""
c=0
while (c<len(doc)): 
  if( doc[c+1] == "opcional" and doc[c]=="pasaporte"):
   c=c+2
   while (c<len(doc)):
    if( doc[c] == "2.1"):
     c=len(doc)
    else: 
     hecho.append(doc[c])
     nombrecedula2=nombrecedula2+doc[c]+" "
     c=c+1
  c=c+1

#print(nombrecedula2)

########################## DATA SET SENTENCIAS Y DIVISIÓN TRAINING Y TEST 70 - 30 

import pandas as pd

df = pd.read_table('sentencias2.csv', 
                   sep=';', 
                   names=['label','sms_message'],encoding='iso-8859-1')
documents = df
# Importar el contador de vectorizacion e inicializarlo
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()

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

################### ALGORIMTO NAIVE BAYES ##################################33
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
tt=testing_data
predictions = naive_bayes.predict(testing_data)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


########### PREDICCION CON LO EXTRAIDO DEL PDF - DEMANDA ##########################3 
X_test1=[demanda]
testing_data = count_vector.transform(X_test1)
predictionsNB = naive_bayes.predict(testing_data)
predicNB = naive_bayes.predict(tt)

##EVALUACIONES
ANB,PNB,RNB,FNB=vali(y_test, predicNB,ngrup)

############################################# SVM #################################################################

from sklearn import svm
X = training_data
y = y_train
clf = svm.SVC()
clf.fit(X, y)
predictionsSVM=clf.predict(testing_data)
predicSVM = clf.predict(tt)
ASVM,PSVM,RSVM,FSVM=vali(y_test, predicSVM,ngrup)

############################################ RL ###################

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0,solver='lbfgs',class_weight='balanced', max_iter=10000).fit(X, y)
predictionsRL=clf.predict(testing_data)
predicRL = clf.predict(tt)
##EVALUACIONES
ARL,PRL,RRL,FRL=vali(y_test, predicRL,ngrup)

########################################### KNN ####################

from sklearn.neighbors import NearestCentroid
import numpy as np
clf = NearestCentroid()
clf.fit(X, y)
predictionsKNN=clf.predict(testing_data)
predicKNN = clf.predict(tt)
##EVALUACIONES
AKNN,PKNN,RKNN,FKNN=vali(y_test, predicKNN,ngrup)


######## CONDICION CON REGRESION LOGISTICA POR TENER EL MEJOR ACCURACY #######
if(predictionsRL==1):
 res='GANA'

else:
 res='PIERDE'


######## PREPARACION DE DATOS PARA ENVIAR A NODEJS ###################
predictionsNB=''.join(map(str, predictionsNB))
predictionsSVM=''.join(map(str, predictionsSVM))
predictionsRL=''.join(map(str, predictionsRL))
predictionsKNN=''.join(map(str, predictionsKNN))
demand=''.join(map(str,demanda))
masdata={
"nombrecedula":nombrecedula,
"celular":celular,
"correo":correo,
"nombrecedula2":nombrecedula2,
"NB":predictionsNB,
"ANB":ANB,
"PNB":PNB,
"RNB":RNB,
"FNB":FNB,
"SVM":predictionsSVM,
"ASVM":ASVM,
"PSVM":PSVM,
"RSVM":RSVM,
"FSVM":FSVM,
"RL":predictionsRL,
"ARL":ARL,
"PRL":PRL,
"RRL":RRL,
"FRL":FRL,
"KNN":predictionsKNN,
"AKNN":AKNN,
"PKNN":PKNN,
"RKNN":RKNN,
"FKNN":FKNN,
"DEM":demand,
"res":res

}




print(json.dumps(masdata))
#print(json.dumps(celular))
#print(json.dumps(correo))
#print(json.dumps(nombrecedula2))
#print(json.dumps(demanda))
#print(json.dumps(res))
sys.stdout.flush()
