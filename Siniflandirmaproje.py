

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#2.1.veri yukleme 

veriler = pd.read_excel('Iris.xls')
print(veriler)


#Bağımlı ve bağımsız değişkenlerin belirlenmesi
x = veriler.iloc[:,:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(x)
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Tek tek algoritmaların denenmesi
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

#Lineer Regresyon
cm = confusion_matrix(y_test,y_pred)
print('LR')
print(cm)


#K-NN Algoritması
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('K-NN')
print(cm)


#Support Vector Machine : Sınıflandırma için kullanılan bir yöntemdir.
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)


#Gaussian Naive Bayes Sınıflandırma
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

#Decision Tree Classifier --> Karar Ağacı Sınıflandırma problemlerin çözümünde yaygın olarak kullanılır.
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

#Random Forest Classifier --> Birden fazla karar ağacı kullanılır. Daha isabetli sınıflandırma yapmaya çalışan sınıflandırma Modelidir.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)

#Bu çıktıların sonunda algoritmaları analiz ettiğimizde K-NN, SVC, DTC, RFC doğruya en yakın sonucu verenler algoritmalardır.
#En iyi sonucu veren ise SVC Algoritması olmuştur. Burada SVC algoritmasının kullanılması daha iyi olacaktır.

    







