# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns import cufflinks as cf %matplotlib inline cf.go_offline()

titan=pd.read_csv('drive/MyDrive/Data Science/titanic_dataset.csv')

titan.head()


<img width="1373" height="242" alt="image" src="https://github.com/user-attachments/assets/bd81f207-ee4f-4453-bbda-0456afc27a1a" />
titan.isnull()
<img width="989" height="478" alt="image" src="https://github.com/user-attachments/assets/da1180e2-9b09-43c5-946c-08a72fb36035" />
sns.heatmap(titan.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')
<img width="736" height="587" alt="image" src="https://github.com/user-attachments/assets/a1148d27-2bef-4c21-8af7-56f2449df22a" />
sns.set_style('whitegrid') sns.countplot(x='Survived',data=titan,palette='RdBu_r')
<img width="1586" height="664" alt="image" src="https://github.com/user-attachments/assets/9fe56ae1-7a7e-4695-8328-0eac789359e9" />
sns.set_style('whitegrid') sns.countplot(x='Survived',hue='Sex',data=titan,palette='RdBu_r')
<img width="991" height="524" alt="image" src="https://github.com/user-attachments/assets/8cad54d1-caee-4abc-9a25-b41de3c97991" />
sns.set_style('whitegrid') sns.countplot(x='Survived',hue='Pclass',data=titan,palette='rainbow')
<img width="777" height="521" alt="image" src="https://github.com/user-attachments/assets/7493ff6a-0d20-4b63-a50e-8362d98e4435" />
sns.displot(titan['Age'].dropna(),kde=False,color='darkred',bins=40)
<img width="700" height="575" alt="image" src="https://github.com/user-attachments/assets/9f33db52-63db-49f6-aff1-4d45679c5c2f" />
titan['Age'].hist(bins=30,alpha=0.3)
<img width="743" height="508" alt="image" src="https://github.com/user-attachments/assets/d8f5fb40-fa36-46a3-9659-40f66656a861" />
sns.countplot(x='SibSp',data=titan)
<img width="837" height="518" alt="image" src="https://github.com/user-attachments/assets/50726aa0-c716-402c-be61-c6d35f8c23be" />
titan['Fare'].hist()
<img width="764" height="498" alt="image" src="https://github.com/user-attachments/assets/6b317b4c-a103-4802-b9e9-d9b012da1e45" />
plt.figure(figsize=(12,7)) sns.boxplot(x='Pclass',y='Age',data=titan,palette='winter')
<img width="715" height="592" alt="image" src="https://github.com/user-attachments/assets/d71ce24b-931c-4b84-b3de-a5a2a777851a" />
def impute_age(cols): Age=cols[0] Pclass=cols[1] if pd.isnull(Age): if Pclass == 1: return 37 elif Pclass == 2: return 29 else: return 24 else: return Age

titan['Age'] = titan[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(titan.isnull(),yticklabels=False,cbar=False,cmap='viridis')
<img width="1263" height="242" alt="image" src="https://github.com/user-attachments/assets/60fb0291-32f7-4b42-bae5-c90aa9fce2a5" />
titan.drop('Cabin',axis=1,inplace=True)

titan.head()
<img width="456" height="368" alt="image" src="https://github.com/user-attachments/assets/e8e937a8-68be-4487-ba77-63074527822a" />
titan.dropna(inplace=True)

titan.info()
<img width="198" height="230" alt="image" src="https://github.com/user-attachments/assets/e837d18e-c69e-4eb7-b295-15b83d2d442a" />
pd.get_dummies(titan['Embarked'],drop_first=True).head()
<img width="601" height="239" alt="image" src="https://github.com/user-attachments/assets/aaa70510-f1ae-4a44-a303-49c1cedc76e9" />
sex=pd.get_dummies(titan['Sex'],drop_first=True) embark=pd.get_dummies(titan['Embarked'],drop_first=True)

titan.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

titan.head()
<img width="760" height="240" alt="image" src="https://github.com/user-attachments/assets/8f4f3aed-e632-4616-95a6-80a0f9b480e5" />
titan=pd.concat([titan,sex,embark],axis=1)

titan.head()


<img width="677" height="232" alt="image" src="https://github.com/user-attachments/assets/115902fe-c05f-4f89-ad77-2bed74a1c789" />
titan.drop('Survived',axis=1).head()
<img width="229" height="270" alt="image" src="https://github.com/user-attachments/assets/d39e898b-05a7-4397-b701-89cade6b5380" />
titan['Survived'].head()
<img width="296" height="55" alt="image" src="https://github.com/user-attachments/assets/4a03d6b9-a2e4-41d2-b0eb-35db197a3f14" />
from sklearn.model_selection import train_test_split

X_titan,X_test,Y_titan,Y_test = train_test_split(titan.drop('Survived',axis=1),titan['Survived'],test_size=0.30,random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression() logmodel.fit(X_titan,Y_titan)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(Y_test,predictions)

accuracy


<img width="741" height="259" alt="image" src="https://github.com/user-attachments/assets/12c4e597-22df-4c5b-9196-2eb67a7454db" />
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y_test,predictions) accuracy

predictions
# RESULT
     Data analysis was completed successfully
