## compute_input.py

import sys, json,requests, numpy as np
#-- coding: iso-8859-1 --
import fitz
import re
from tabula import read_pdf
import tabula
import pandas as pd
import numpy as np
import csv


carp="public"
pdf= sys.argv[1]
pdf=carp+pdf
df= read_pdf(pdf, pages="2")
#print(df)

tabula.convert_into(pdf,("frmXYelim-291272" + '.csv'),output_format="csv",pages="2")

pdf_documento="f1.pdf"
documento= fitz.open(pdf_documento)
#print("NUnmero de pag: ",documento.pageCount)
#print("Metadatos: ",documento.metadata)
pagina = documento.loadPage(1)
text= pagina.getText("text")
doc=(re.sub('[!?@#$()-.,;:*/0-9%"]+',' ',text.lower())).split()
#print(docS)
#doc=[]
#for m in docS:
# doc.append(normalize(m))

c=0
m=0
hecho=[]
vectorh=""
while (c<len(doc)): 
  if( doc[c] == "artículos"):
   c=c+1
   while (c<len(doc)):
    if( doc[c] == "sumario"):
     c=len(doc)
    else: 
     hecho.append(doc[c])
     vectorh=vectorh+doc[c]+" "
     c=c+1
  c=c+1


#print(vectorh)
import pandas as pd

df = pd.read_table('sentencias2.csv', 
                   sep=';', 
                   names=['label','sms_message'],encoding='iso-8859-1')
# Visualización de las 5 primeras filas
#print(df.head())


# Conversion
#df['label'] = df.label.map(('neg':0), ('pos':1))
# Visualizar las dimensiones de los datos
#print(df.shape())

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

#print(X_train)



#print('Number of rows in the total set: {}'.format(df.shape[0]))
#print('Number of rows in the training set: {}'.format(X_train.shape[0]))
#print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# Instantiate the CountVectorizer method
count_vector = CountVectorizer()
# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
#print('Precision score: ', format(precision_score(y_test, predictions)))
#print('Recall score: ', format(recall_score(y_test, predictions)))
#print('F1 score: ', format(f1_score(y_test, predictions)))

#print("-----------------------------------------------")

#nega="el señor no es padre biologico de mi hijo"
#vectorh=vectorh+nega
X_test1=[vectorh]
#print(X_test1)

testing_data = count_vector.transform(X_test1)
#print("---testing_data = ",testing_data)

predictions = naive_bayes.predict(testing_data)
#print("---prediccion = ",predictions)

if(predictions==1):
 res="GANA"
 #print(res)
else:
 res="PIERDE"
 #print(res)  





resp=res

print(json.dumps(resp))
sys.stdout.flush()