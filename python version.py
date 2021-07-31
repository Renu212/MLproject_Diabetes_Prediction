# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # **DIABETES PREDICTION**
# %% [markdown]
# <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSNaOH_nV4JyQkwc34EDVlqXr6JQZGxUSSt4A&usqp=CAU" >
# %% [markdown]
# The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. The given dataset female gender and Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# %% [markdown]
# ## Data description
# %% [markdown]
#  > Pregnancies               -     Number of times pregnant <br>
#  > Glucose                   -     Plasma glucose concentration a 2 hours in an oral glucose tolerance test<br>
#  > Blood Pressure            -     Diastolic blood pressure (mm Hg)<br>
#  > Skin Thickness            -     Triceps skin fold thickness (mm)<br>
#  > Insulin                   -     2-Hour serum insulin (mu U/ml)<br>
#  > BMI                       -     Body mass index (weight in kg/(height in m)^2)<br>
#  > DiabetesPedgreeFunction   -     Diabetes pedgree function<br>
#  > Age                       -     Age(years)<br>
#  > outcome                   -     class variable(0 or 1)<br>
# %% [markdown]
# <h2 style='background:skyblue; border:0px; color:black'><center><strong>Dataset import and Feature engineering<strong><center><h2>

# %%
import pandas as pd
import numpy as np

import opendatasets as od

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9,4 )
matplotlib.rcParams['figure.facecolor'] = '#00000000'



# %%
dataset_url = "https://www.kaggle.com/uciml/pima-indians-diabetes-database"
od.download(dataset_url)


# %%
data_dir="./pima-indians-diabetes-database"
os.listdir(data_dir)


# %%
data = pd.read_csv("./pima-indians-diabetes-database/diabetes.csv")
data.head()


# %%
data.info()


# %%
data["Outcome"].value_counts()


# %%

fig,ax = plt.subplots()
#plt.rcParams['font.sans-serif'] = 'Arial'
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['text.color'] = 'white'
#plt.rcParams['axes.labelcolor']= 'white'
#plt.rcParams['xtick.color'] = 'white'
#plt.rcParams['ytick.color'] = 'white'
#plt.rcParams['font.size']=12
labels = ['Diabetic','Non-Diabetic']
percentages = [(500/(768))*100, (268/(768))*100]
explode = (0.1,0)
ax.pie(percentages,explode=explode,labels=labels, autopct='%1.0f%%', 
       shadow=False, startangle=45,   
       pctdistance=1.2,labeldistance=1.5)
ax.axis('equal')
ax.legend(frameon=False, bbox_to_anchor=(1.4,0.8))


# %%
min_values=data.describe().iloc[3,:]
lst = list(min_values.index)
for i in lst:
    print("Minimum value of  column {} is {} ".format(i,min_values[i]))

# %% [markdown]
# Glucose,Blood Pressure,Skin thickness, Insulin,BMI cannot have values 0.<br>
# Hence we can replace this data into NAN values
# 

# %%
data.isnull().sum()


# %%
columns = ['Glucose','BloodPressure','SkinThickness', 'Insulin','BMI']
for i in columns:
    data[i].replace(0,np.nan,inplace=True)
k=data.isnull().sum()
sns.countplot(k)
plt.xlabel(columns)
plt.rcParams['text.color'] = 'black'


# %%
data.isnull().sum()


# %%
columns1 = ['Pregnancies','Glucose','BloodPressure','SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age']
sns.set(rc={'figure.figsize':(10,17)})

colours=['r','c','k','m','r','y','b','g']

for i in range(len(columns1)):
    
    plt.subplot(4,2,i+1)
    sns.distplot(data[columns1[i]], hist=True, rug=True, color=colours[i])


# %%
columns1 = ['Pregnancies','Glucose','BloodPressure','SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age']
sns.set(rc={'figure.figsize':(10,17)})

colours=['r','c','k','m','r','y','b','g']

for i in range(len(columns1)):
    
    plt.subplot(4,2,i+1)
    sns.boxplot(data[columns1[i]],  color=colours[i])


# %%
from scipy.stats import skew
for i in columns1:
    print('Skewness of {} is {}'.format(i,data[i].skew()))
    

# %% [markdown]
# It is observed that column Insulin and Diabete Pedigree Function data is highly skewed
# %% [markdown]
# From the above boxplots and distribution plots, it is observed that the data related to insulin is skewed and have more outliers. Outlier data points will have impact on mean ,hence, in such cases, choosing median as a imputer is a good choice. For symmetric distribution ,mean can be used as an imputer technique for missing values.
# 
# Hence, i would like to use median for replacing null values in insulin data column and mean for other columns containing null values.

# %%
# Impute missing numerical values
# median is used to replace null in insulin column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
data['Insulin']= imputer.fit_transform(data[['Insulin']])
data['Insulin'].isnull().any()


# %%
# mean is used to replace null in other columns
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
columns2 = ['Glucose','BloodPressure','SkinThickness','BMI']

data[columns2]= imputer.fit_transform(data[columns2])
data[columns2].isnull().any()

# %% [markdown]
# ## EDA

# %%

sns.set(rc={'figure.figsize':(10,17)})

#colours=['r','c','k','m','r','y','b','g']

for i in range(len(columns1)):
    
    plt.subplot(4,2,i+1)
    sns.boxplot(x=data['Outcome'],y=data[columns1[i]],hue=data['Outcome'])

# %% [markdown]
# It can be observed that:
# 
# Diabetic people have higher DiabetePedifreeFunction i.e, genetic influence plays some role in diabetes among patients
# 
# Higher glucose levels leads to more chances of Diabetes
# 
# When the Blood pressure is higher there are chances of having diabetes
# 
# Higher the insulin more the chances of diabetes
# 
# less chance of diabetes among young people
# 
# Higher the BMI, more is risk of having diabetes

# %%
sns.set(rc={'figure.figsize':(9,4)})
sns.stripplot(x='Outcome',y='Pregnancies',data=data)

# %% [markdown]
# we can see higher number of pregnancies result in having diabetes
# %% [markdown]
# Pair Plots are a really simple (one-line-of-code simple!) way to visualize relationships between each variable. It produces a matrix of relationships between each variable in your data for an instant examination of our data.

# %%

sns.pairplot(data,hue='Outcome',palette='magma')

# %% [markdown]
# correlation Matrix

# %%
sns.set(rc={'figure.figsize':(20,10)})

sns.heatmap(data.corr(),annot=True,cmap='viridis')

# %% [markdown]
# It is observed that the features are low corelated, hence less chance of mutlicollinearity
# %% [markdown]
# ## Data preprocessing

# %%
input_cols = list(data.columns)[0:8]
input_df = data[input_cols].copy()
target_df = data["Outcome"]
input_df[input_cols]


# %%
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
input_df= pd.DataFrame(sc.fit_transform(input_df),columns=input_cols)


# %%
# splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(input_df,target_df,test_size=0.30,random_state=42)
x_train

# %% [markdown]
# ## **MODEL SELECTION**

# %%
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import accuracy_score

# %% [markdown]
# <h2 style='background:skyblue; border:0px; color:black'><center><strong>Logistic Regression<strong><center><h2>

# %%
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(penalty='l2',C=1,solver='liblinear')
lr_model.fit(x_train,y_train)

print("Train Set Accuracy:",accuracy_score(y_train,lr_model.predict(x_train))*100)
print("Test Set Accuracy:",accuracy_score(y_test,lr_model.predict(x_test))*100)

# %% [markdown]
# ## Using GridSearch CV

# %%
log_params = {'penalty':['l1', 'l2'], 
              'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 100], 
              'solver':['liblinear', 'saga']} 
log_model = GridSearchCV(LogisticRegression(), log_params,n_jobs=-1, cv=5) #Tuning the hyper-parameters
log_model.fit(x_train, y_train)
log_predict = log_model.predict(x_test)
log_score = log_model.best_score_
print('---------------------------------')
print('Best parameters: ',log_model.best_params_)
print('---------------------------------')
print('Best score: ', log_score*100)



# %%
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = log_model.predict(x_test)
confusion_matrix(y_test,y_pred)

# %% [markdown]
# 
# <h2 style='background:skyblue; border:0px; color:black'><center><strong>Decision Tree<strong><center><h2>

# %%
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion='entropy',max_depth=5,splitter='best')
dt_model.fit(x_train,y_train)



print("Train Set Accuracy:",accuracy_score(y_train,dt_model.predict(x_train))*100)
print("Test Set Accuracy:",accuracy_score(y_test,dt_model.predict(x_test))*100)

# %% [markdown]
# ## Using GridSearch CV

# %%

from sklearn.tree import DecisionTreeClassifier
dtc_params = {'criterion' : ['gini', 'entropy'],
              'splitter': ['random', 'best'], 
              'max_depth': [3, 5, 7, 9, 11, 13]}
dtc_model = GridSearchCV(DecisionTreeClassifier(), dtc_params, cv=5,n_jobs=-1) #Tuning the hyper-parameters
dtc_model.fit(x_train, y_train)
dtc_predict = dtc_model.predict(x_test)
dtc_score = dtc_model.best_score_
print('---------------------------------')
print('Best parameters: ',dtc_model.best_params_)
print('---------------------------------')
print('Best score: ', dtc_score*100)



# %%
y_pred = dtc_model.predict(x_test)
confusion_matrix(y_test,y_pred)

# %% [markdown]
# <h2 style='background:skyblue; border:0px; color:black'><center><strong>Random Forest<strong><center><h2>

# %%
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(criterion='entropy',max_depth=5,n_estimators= 5)
rf_model.fit(x_train,y_train)



print("Train Set Accuracy:",accuracy_score(y_train,rf_model.predict(x_train))*100)
print("Test Set Accuracy:",accuracy_score(y_test,rf_model.predict(x_test))*100)

# %% [markdown]
# ## Using GridSearch CV

# %%
rfc_params = {'criterion' : ['gini', 'entropy'],
             'n_estimators': list(range(5, 26, 5)),
             'max_depth': list(range(3, 20, 2))}
rfc_model = GridSearchCV(RandomForestClassifier(), rfc_params, cv=5) #Tuning the hyper-parameters
rfc_model.fit(x_train, y_train)
rfc_predict = rfc_model.predict(x_test)
rfc_score = rfc_model.best_score_
print('---------------------------------')
print('Best parameters: ',rfc_model.best_params_)
print('---------------------------------')
print('Best score: ', rfc_score*100)


# %%
y_pred = rfc_model.predict(x_test)
confusion_matrix(y_test,y_pred)

# %% [markdown]
# <h2 style='background:skyblue; border:0px; color:black'><center><strong>Support Vector Classification<strong><center><h2>

# %%
from sklearn.svm import SVC
svm_model = SVC(C=1,kernel='linear')
svm_model.fit(x_train,y_train)    



print("Train Set Accuracy:",(accuracy_score(y_train,svm_model.predict(x_train))*100))
print("Test Set Accuracy:",(accuracy_score(y_test,svm_model.predict(x_test))*100))

# %% [markdown]
# ## Using GridSearch CV

# %%
svc_params = {'C': [0.001, 0.01, 0.1, 1],
              'kernel': [ 'linear' , 'poly' , 'rbf' , 'sigmoid' ]}
svc_model = GridSearchCV(SVC(), svc_params, cv=5) #Tuning the hyper-parameters
svc_model.fit(x_train, y_train)
svc_predict = svc_model.predict(x_test)
svc_score = svc_model.best_score_

print('---------------------------------')
print('Best parameters: ',svc_model.best_params_)
print('---------------------------------')
print('Best score: ', svc_score*100)


# %%
y_pred = svc_model.predict(x_test)
confusion_matrix(y_test,y_pred)

# %% [markdown]
# <h2 style='background:skyblue; border:0px; color:black'><center><strong>KNearest Neighbors<strong><center><h2>

# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9,weights='uniform',algorithm='kd_tree',metric='euclidean')                #knn classifier
knn.fit(x_train,y_train)

knn_acc = accuracy_score(y_test,knn.predict(x_test))


print("Train Set Accuracy:",(accuracy_score(y_train,knn.predict(x_train))*100))
print("Test Set Accuracy:",(accuracy_score(y_test,knn.predict(x_test))*100))

# %% [markdown]
# ## Using GridSearch CV

# %%
knn_params = {'n_neighbors': list(range(3, 20, 2)),
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
          'metric':['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5) #Tuning the hyper-parameters
knn_model.fit(x_train, y_train)
knn_predict = knn_model.predict(x_test)
knn_score = knn_model.best_score_

print('---------------------------------')
print('Best parameters: ',knn_model.best_params_)
print('---------------------------------')
print('Best score: ', knn_score*100)


# %%
y_pred = knn_model.predict(x_test)
confusion_matrix(y_test,y_pred)

# %% [markdown]
# ## Conclusion

# %%
models = ['LogisticRegression', 'KNeighborsClassifier', 'SVC', 'DecisionTreeClassifier', 
          'RandomForestClassifier']
scores = [log_score, knn_score, svc_score,dtc_score,rfc_score]
score_table = pd.DataFrame({'Model':models, 'Score':scores})
score_table.sort_values(by='Score', axis=0, ascending=False)
print(score_table.sort_values(by='Score', ascending=False))
sns.barplot(x = score_table['Score'], y = score_table['Model'], palette='inferno');



