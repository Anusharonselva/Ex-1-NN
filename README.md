<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
ENTER YOUR NAME : S.ANUSHARON
ENTER YOUR REGISTER NO : 212222240010
```
```
#import libraries

import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
d=pd.read_csv("Churn_Modelling.csv")
df=pd.DataFrame(d)
d.head()
d

#Finding Missing Values
print(d.isnull().sum())

d.info()
d.drop(['Surname', 'Geography','Gender'], axis=1)

#Check for Duplicates
print(d.duplicated().sum())

#Detect Outliers
#Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = d.quantile(0.25)
Q3 = d.quantile(0.75)

#Calculate the IQR
IQR = Q3 - Q1

#Normalize the dataset
#Create an instance of MinMaxScaler
scaler = MinMaxScaler()

#Define the columns to be normalized
columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

#Normalize the specified columns
d[columns] = scaler.fit_transform(d[columns])

#Display the normalized dataset
print("NORMALIZED DATASET\n",d)

#split the dataset into input and output
X = d.iloc[:,:-1].values
print("INPUT(X)\n",X)
y = d.iloc[:,-1].values
print("OUTPUT(y)\n",y)


#splitting the data for training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("X_train\n")
print(X_train)
print("\nX_test\n")
print(X_test)
print("\nY_train\n")
print(y_train)
print("\nY_test\n")
print(y_test)
```

## OUTPUT:
d.head()
![Screenshot 2024-02-29 231340](https://github.com/Anusharonselva/Ex-1-NN/assets/119405600/1ed56538-c4c4-4144-ba09-9816749b2c6f)
X Values:
![Screenshot 2024-02-29 231706](https://github.com/Anusharonselva/Ex-1-NN/assets/119405600/5cfb082a-691b-4f79-ad91-df0d2c5d2217)
Y Values:
![Screenshot 2024-02-29 231742](https://github.com/Anusharonselva/Ex-1-NN/assets/119405600/a9c7c36a-334a-4861-bae3-75c5930be141)
Null Values:
![Screenshot 2024-02-29 231823](https://github.com/Anusharonselva/Ex-1-NN/assets/119405600/88e7add7-92df-481b-a89d-491a2895546a)
Duplicated Values:
![Screenshot 2024-02-29 231907](https://github.com/Anusharonselva/Ex-1-NN/assets/119405600/7f1be0bd-eade-40b0-b903-e6f1a861c7fe)
Description:
![Screenshot 2024-02-29 231949](https://github.com/Anusharonselva/Ex-1-NN/assets/119405600/c974abb9-d27d-4dcc-b31c-c454fc27d4b8)
Normalized Dataset:
![Screenshot 2024-02-29 232054](https://github.com/Anusharonselva/Ex-1-NN/assets/119405600/e0dd8959-331e-4297-be8e-36180f469dbd)
Training Data:
![Screenshot 2024-02-29 232201](https://github.com/Anusharonselva/Ex-1-NN/assets/119405600/1cf17076-60ef-40ca-833a-d88e419252cd)
X_test:
![Screenshot 2024-02-29 232211](https://github.com/Anusharonselva/Ex-1-NN/assets/119405600/1f900fef-554e-4392-aaae-4ed2798adf17)
length :
![Screenshot 2024-02-29 232331](https://github.com/Anusharonselva/Ex-1-NN/assets/119405600/1ddea0a8-4e8a-4a2b-8820-3488c6edda0d)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


