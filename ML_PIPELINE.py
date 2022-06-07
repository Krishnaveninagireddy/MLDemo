
"""
Created on Mon Jun  6 15:51:21 2022

@author: krish
"""

import numpy as np
import pandas as pd

training_data = pd.read_csv('Sample_modelCSV.csv')
training_data.describe()

x= training_data.iloc[:, :-1].values
y=training_data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.20,random_state=0)

"""
from sklearn.preprocessing import standardscaler
sc= standardscaler()
x_train = sc.fit_transform(x_train)
x_test=sc.transform(x_test)
"""

"""#build classificaion model
### we azre using KNN classifier in this example
*n_neighbors = 5 - *number of neghbors
*metric = 'minkwoski',p=2* - for eduxation distance calculation
"""

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test=sc.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier
# minkowski is for ecledian distance
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

#model trainig
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
y_prob = classifier.predict_proba(x_test)[:,1]

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

new_prediction =  classifier.predict(sc.transform(np.array([[40,20000]])))

new_prediction_proba =  classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]

new_predict = classifier.predict(sc.transform(np.array([[42,50000]])))


#pickling the Model and Standard Scaler
#pickle --To serialize the a python object/to convert byte stream
import pickle
model_file = "classifier_pickle"
pickle.dump(classifier,open(model_file,'wb'))

scaler_file="sc.pickle"
pickle.dump(sc, open(scaler_file, 'wb'))
