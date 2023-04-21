# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: D.vishnu vardhan redddy
RegisterNumber:  212221230023
```
```
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
1.Placement data

![image](https://user-images.githubusercontent.com/94175324/233582087-b71260ee-e585-40cb-90b8-2c63eb548e75.png)


2.Salary data

![image](https://user-images.githubusercontent.com/94175324/233582173-27c2c854-b39d-47e7-ba72-1608908becdb.png)


3.Checking the null() function

![image](https://user-images.githubusercontent.com/94175324/233582761-fe98c513-0324-4a0f-81f6-3be95ef66690.png)


4. Data Duplicate

![image](https://user-images.githubusercontent.com/94175324/233582835-1bee8552-917a-4a31-93bf-3edddfc55023.png)


5. Print data

![image](https://user-images.githubusercontent.com/94175324/233583999-869a5f1a-fd0b-475a-acb9-231b56ca592d.png)


6. Data-status

![image](https://user-images.githubusercontent.com/94175324/233584803-4a0828c8-ca5d-4321-a592-3efeb78b856e.png)


![image](https://user-images.githubusercontent.com/94175324/233584920-3b2a1583-f16b-46bf-8f3a-0bf481a2c7ae.png)



7. y_prediction array

![image](https://user-images.githubusercontent.com/94175324/233584985-3e7e24fc-d9eb-415d-83eb-75fbeb33f941.png)


8.Accuracy value

![image](https://user-images.githubusercontent.com/94175324/233585084-74d9dd6f-7900-4668-9ef3-b98e9a740039.png)



9. Confusion array

![image](https://user-images.githubusercontent.com/94175324/233585145-c2c8406c-1344-4fcf-bdf7-de5c966c88f9.png)


10. Classification report

![image](https://user-images.githubusercontent.com/94175324/233585216-eff5a0cd-a3b9-45b0-862e-3ea38753701a.png)



11.Prediction of LR

![image](https://user-images.githubusercontent.com/94175324/233585533-49776a3f-288d-4cef-ad02-f2706df1d9d7.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
