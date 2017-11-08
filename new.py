import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#Read from train.csv
data=pd.read_csv("train.csv").as_matrix()

#Declare a Decision Tree Classifier Object
clf = DecisionTreeClassifier()

#training data
xtrain=data[16000:,1:]
train_label=data[16000:,0]

#testing data
xtest=data[0:21000,1:]
actual_label=data[0:21000:,0]

#training the classifier
clf.fit(xtrain,train_label)

#Calculating accuracy
p=clf.predict(xtest)
count=0
for i in range(0,21000):
    if p[i]==actual_label[i]:
        count+=1

#sample test data (csv-2) , We now predict an image
d=xtest[47]
d.shape=(28,28)
pt.imshow(255-d,cmap='gray')
print("With an Accuracy of : ",(count/21000)*100)
print("The Number is : ",clf.predict( [xtest[47]] ))
pt.show()
