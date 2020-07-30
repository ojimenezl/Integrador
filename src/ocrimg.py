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
from datetime import datetime
now = datetime.now()


carp="public"
img= sys.argv[1]
img=carp+img

#print(df)

#-*- coding: utf-8 -*-
from PIL import Image
import pytesseract
import cv2
import os
import time
import sys
import fitz
imagePath = img
preprocess="thresh"


image= cv2.imread(imagePath)
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if(preprocess=="thresh"):
    gray = cv2.threshold(gray,0,450,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif(preprocess=="blur"):
    gray = cv2.medianBblur(gray,3)

filename= "extracciones/{}.jpg".format(os.getpid())
cv2.imwrite(filename,gray)

#Creación Archivo
outfile='extracciones/'+str(now.day)+'-'+str(now.month)+'-'+str(now.year)+'-'+str(now.hour)+'-'+str(now.minute)+'.txt' 
f = open(outfile,"w") 
img= Image.open(filename,mode = 'r')
#C:\Program Files\
pytesseract.pytesseract.tesseract_cmd=r"Tesseract-OCR\tesseract.exe"
text = pytesseract.image_to_string(img)


masdata={"text":text} 
print(json.dumps(masdata))

sys.stdout.flush()

#Guarda el Texto Reconocido en extraccion.txt
f.write(text)
f.close()
#cv2.imshow("Image",image)
#cv2.imshow("Output",gray)










