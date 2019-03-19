# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 08:51:43 2019

@author: Ahmed Khaled
for multi_linear regression are :
    step 1 :import libararies
    step 2 :Get data set
    step 3 :split data into input & output
    step 4 :check missing data
    step 5: check categeorical data
    step 6 :split data into training data  & test data 
    step 7 :Build your model
    step 8 :plot best line 
    step 9 :Estimate Error

"""

# step 1 :import libararies
import numpy as np   # to make  mathmatical operation on metrices
import pandas as pd  #to read data
import matplotlib.pyplot as plt   #to show some graghs
import seaborn as sns    #for plot data
from sklearn.cross_validation import  train_test_split #to split data to train & test
from sklearn.linear_model import LinearRegression    #to import linear model 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder   #for categeorical data
from sklearn.metrics import mean_squared_error #to calculate MSE ,MAE ,RMSE


# step 2 :Get data set
path1 = 'C:\\Users\\Ahmed Khaled\Downloads\\my work (regression)\\7)house-prices-advanced-regression-techniques\\train.csv'
path2 = 'C:\\Users\\Ahmed Khaled\Downloads\\my work (regression)\\7)house-prices-advanced-regression-techniques\\test.csv'
x = pd.read_csv(path1)  # train data
y = pd.read_csv(path1)   #test data
print('data : \n ',x)
print('test_data : \n ',y)
print('data.head : \n ',x.head())
print('data.shape : \n',x.shape)
print('names of columns :\n',x.columns)
print('data.imnformation: \n ' ,x.info())
print('data.describe: \n ' ,x.describe())
#stats of the predictor variable (saleprice)
print('data.saleprice.describe: \n ' ,x.SalePrice.describe())
#sns.pairplot(x)
sns.distplot(x['SalePrice']) #distrebution
x.corr()    #corrolations

sns.heatmap(x.corr(),annot=True) #relation with number
sns.heatmap(x.corr(),linewidth = 0.2, vmax=1.0, square=True, linecolor='red',annot=True)

#seeking only the numeric features from the data
numeric_features = x.select_dtypes(include = [np.number])
numeric_features.dtypes

#features with the most correlation with the predictor variable
corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending = False)[:6], '\n')
print(corr['SalePrice'].sort_values(ascending = False)[-5:])

x.OverallQual.unique()

#checking the null values
nulls = pd.DataFrame(x.isnull().sum().sort_values(ascending = False)[:25])
nulls.columns = ['Null Count']   #name of columns
nulls.index.name = 'Feature'    #name of index
nulls[:5]
print('Unique values are:', x.MiscFeature.unique())

#analysing the categorical data
categoricals = x.select_dtypes(exclude= [np.number])
categoricals.describe()
print ("Original: \n") 
print (x.Street.value_counts(), "\n")

#One-hot encoding to convert the categorical data into integer data
x['enc_street'] = pd.get_dummies(x.Street, drop_first= True)
y['enc_street'] = pd.get_dummies(y.Street, drop_first= True)

print('Encoded: \n')
print(x.enc_street.value_counts())

#Analysing the feature - Sale Condition
condition_pivot = x.pivot_table(index= 'SaleCondition', values= 'SalePrice', aggfunc= np.median)

condition_pivot.plot(kind= 'bar', color = 'blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation = 0)
plt.show()

#Encode function
def encode(x): 
    if x == 'Partial':
        return 1
    else:
        return 0
#Treating partial as one class and other all sale condition as other
x['enc_condition'] = x.SaleCondition.apply(encode)
y['enc_condition'] = y.SaleCondition.apply(encode)

condition_pivot = x.pivot_table(index= 'enc_condition', values= 'SalePrice', aggfunc= np.median)

condition_pivot.plot(kind= 'bar', color = 'blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation = 0)
plt.show()


#Handling the missing values by interpolation
data = x.select_dtypes(include= [np.number]).interpolate().dropna()

#Verifying missing values
sum(data.isnull().sum() != 0)

#log transforming the target variable to improve the linearity of the regression
y = np.log(x.SalePrice)
#dropping the target variable and the index from the training set
x = data.drop(['SalePrice', 'Id'], axis = 1)

# step 6 :split data into training data  & test data
#splitting the data into training and test set

#from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = .33) 
#cols_x = x.shape[1]
#cols_y = y.shape[1]
#x_train , y_train = x.iloc[:,0:-1].values , x.iloc[:, cols_x-1].values
#x_test , y_test = y.iloc[:,0:-1].values , y.iloc[:, cols_x-1].values

#step 7 :Build your model ,Linear regression model
from sklearn import linear_model
model = linear_model.LinearRegression()
#fitting linear regression on the data
model.fit(x_train, y_train)

#Step 7 - Interpreting the Coefficient and the Intercept
y_pred = model.predict(x_test)
plt.scatter(y_test,y_pred)
sns.distplot((y_test-y_pred),bins=50) 

#Step 8 - Interpreting the Coefficient and the Intercept

print(model.coef_)
print(model.intercept_)

#Step 9 - Predict the Score (% Accuracy)
print('Train Score :', model.score(x_train,y_train))
print('Test Score:', model.score(x_test,y_test))

#estimata error
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test,y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

actual_values = y_test
plt.scatter(y_pred, actual_values, alpha= 0.75, color = 'b')

plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.title('Linear Regression Model')
plt.show()

#Gradient boosting regressor model     
from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(n_estimators= 1000, max_depth= 2, learning_rate= .01)
est.fit(x_train, y_train)

y_train_predict = est.predict(x_train)
y_test_predict = est.predict(x_test)

est_train = mean_squared_error(y_train, y_train_predict)
print('Mean square error on the Train set is: {}'.format(est_train))

est_test = mean_squared_error(y_test, y_test_predict)
print('Mean square error on the Test set is: {}'.format(est_test))