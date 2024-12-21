# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM :
 1. Import chardet and pandas
 
 2. Read the dataset
 
 3. Extract first few rows and columns
 
 4. Find accuracy
 

## PROGRAM :
Developed by: PRIYADARSHINI K

RegisterNumber: 24900922 

import chardet 

file='spam.csv'

with open(file, 'rb') as rawdata:

    result = chardet.detect(rawdata.read(100000))
    
print(result)

import pandas as pd 

data = pd.read_csv("spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

x_train=cv.fit_transform(x_train)

x_test=cv.transform(x_test)

from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

print(y_pred)

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

print(accuracy)

## OUTPUT :
![image](https://github.com/user-attachments/assets/f8b8cedf-9ee1-41a4-bc2e-13e2b89f2379)



## RESULT :
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
