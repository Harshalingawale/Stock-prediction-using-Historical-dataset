#!/usr/bin/env python
# coding: utf-8

# # Project Title

# ## Predicting Best Stocks to Invest Using Nifty 50 Data From Past 10 Years.

# # Business Problem Statement 
# Link to Dataset --[click here](https://www.kaggle.com/datasets/setseries/nifty50-stocks-dataset20102021/data?select=Final-50-stocks.csv)

# ### A company aims to predict the future movements of stocks in the Nifty 50 index. Therefore This prediction can help investors of company to make informed decisions about buying, selling, or holding stocks, thereby they can obtimize  their investment strategies and get maximum returns.

# # Importing libraries

# In[2]:


import pandas as pd
import datetime as dt
import matplotlib.pyplot as plot
import seaborn as abc
from feature_engine.outliers import Winsorizer as wins
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as gsv
from sklearn.linear_model import LinearRegression as lnr
from sklearn.ensemble import GradientBoostingRegressor as grd, RandomForestRegressor as randf
from sklearn.svm import SVR as sm
from sklearn.neighbors import KNeighborsRegressor as knr
from sklearn.metrics import mean_squared_error as me_err, mean_absolute_error as ma_err, r2_score as r2_accuracy
import warnings
warnings.filterwarnings('ignore')


# In[3]:


nif50 = pd.read_csv("datasets/Final-50-stocks.csv")


# #### we have loaded the dataset

# # Data Exploration

# In[4]:


nif50.info()


# In[5]:


nif50.dtypes


# In[6]:


nif50.columns


# # Data Processing and Feature Engineering

# In[7]:


nif50['DATE']=pd.to_datetime(nif50.DATE)
nif50['Year']=nif50.DATE.dt.year 
nif50['Month']=nif50.DATE.dt.month
nif50['weekday'] = nif50['DATE'].dt.weekday


# #### In this we change the values in the date, column of the  nif50 into date and time objects. Then we made a new column name month, year in  dataframe of nif50.

# In[8]:


nif50.HEROMOTOCO.isnull().value_counts()


# #### This applies a boolean mask to the column 'HEROMOTOCO', returning a boolean Series in which each element is True if the other value in the column is null, and False.

# In[9]:


nif50.HEROMOTOCO.fillna(nif50.HEROMOTOCO.mean(),inplace=True)
nif50.HEROMOTOCO


# In[10]:


nif50.COALINDIA.isnull().value_counts()
nif50.COALINDIA.fillna(nif50.COALINDIA.mean(),inplace=True)
nif50.COALINDIA


# In[11]:


nif50.UPL.isnull().value_counts()
nif50.UPL.fillna(nif50.UPL.mean(),inplace=True)
nif50.UPL


# In[12]:


nif50.ADANIPORTS.isnull().value_counts()
nif50.ADANIPORTS.fillna(nif50.ADANIPORTS.mean(),inplace=True)
nif50.ADANIPORTS


# In[13]:


nif50.INFY.isnull().value_counts()
nif50.INFY.fillna(nif50.ADANIPORTS.mean(),inplace=True)
nif50.INFY


# #### This replaces missing (NaN) values with a specified value and calculates the mean value of the non-missing values and after that the parameter specifies that the changes should be applied directly to nif50.

# In[14]:


clean_data = nif50.isna().sum().sum()
print("Null Values in dataset - ", clean_data)


# #### Therefore all Null Values are removed.

# # Visualization
# Displot for data distribution for Top 10 stocks

# In[15]:


image_width = 10
image_height = 8

fig, axes = plot.subplots(nrows=3, ncols=3, figsize=(image_width, image_height))
axes = axes.flatten()

for i, ax in zip(range(1, 10), axes):
    ax.violinplot(nif50.iloc[:, i], vert=False)
    ax.set_title(f"Distribution of {nif50.columns[i]}")
    ax.set_xlabel("Stock Price")

plot.tight_layout()
plot.show()


# #### In this we have ploted  10 stocks and created a distribution plot using plotly.

# # Removing Outliers

# In[16]:


all_columns = ['TATASTEEL', 'WIPRO', 'TITAN', 'ULTRACEMO', 'TECHM', 'RELIANCE',
       'SHREECEM', 'SUNPHARMA', 'TATAMOTORS', 'TCS', 'SBIN', 'NESTLEIND',
       'NTPC', 'M&M', 'MARUTI', 'ONGC', 'POWERGRID', 'JSWSTEEL', 'KOTAKBANK',
       'LT', 'ICICIBANK', 'INDUSBANK', 'INFY', 'IOC', 'ITC', 'HEROMOTOCO',
       'HINDALCO', 'HINDUNILVR', 'HCLTECH', 'HDFCBANK', 'HDFC', 'DRREDDYS',
       'EICHERMOTOR', 'GRASIM', 'CIPLA', 'COALINDIA', 'BPCL', 'BRITANNIA',
       'ADANIPORTS', 'BAJAJFINSERV', 'BAJAJFINANCE', 'BHARTIARTL', 'AXISBANK',
       'BAJAJ-AUTO', 'ASIANPAINT', 'UPL']

winsorizer = wins(capping_method='gaussian', tail='both', fold=1.5, variables=all_columns)

nif50 = winsorizer.fit_transform(nif50)


# In[17]:


my_data_remove = nif50.drop(columns=['DATE'])
my_cols = 5

my_rows = (len(my_data_remove.columns) + my_cols - 1) // my_cols
fig, a = plot.subplots(nrows=my_rows, ncols=my_cols, figsize=(20, 15))

a = a.flatten()

for i, column in enumerate(my_data_remove.columns):
    ax = a[i]
    my_data_remove.boxplot(column=column, ax=ax)
    ax.set_title(column)

for j in range(len(my_data_remove.columns), my_rows * my_cols):
    a[j].axis('off')

plot.tight_layout()
plot.show()


# #### box plot first we have filtered the date column then after defineing the number of columns we calculate number of rows then  flatten the axes array to iterate over them and plot box plots for each columns,therefore we have removed the outliers.

# # Training and Testing

# In[28]:


X = nif50[['Year', 'Month', 'weekday']]  

main_variables = ('TATASTEEL', 'WIPRO', 'TITAN', 'ULTRACEMO', 'TECHM', 'RELIANCE',
       'SHREECEM', 'SUNPHARMA', 'TATAMOTORS', 'TCS', 'SBIN', 'NESTLEIND',
       'NTPC', 'M&M', 'MARUTI', 'ONGC', 'POWERGRID', 'JSWSTEEL', 'KOTAKBANK',
       'LT', 'ICICIBANK', 'INDUSBANK', 'INFY', 'IOC', 'ITC', 'HEROMOTOCO',
       'HINDALCO', 'HINDUNILVR', 'HCLTECH', 'HDFCBANK', 'HDFC', 'DRREDDYS',
       'EICHERMOTOR', 'GRASIM', 'CIPLA', 'COALINDIA', 'BPCL', 'BRITANNIA',
       'ADANIPORTS', 'BAJAJFINSERV', 'BAJAJFINANCE', 'BHARTIARTL', 'AXISBANK',
       'BAJAJ-AUTO', 'ASIANPAINT')

for tar_vari in main_variables:
    y = nif50[tar_vari]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### firstly we have add features like year,month,weekdays and then defined target variables and then iterated each main variable and divide  data in train and test sets.

# # Tune Hyperparameters
# Below are the various Machine Learning Algorithms used to build the pipeline

# ## 1.Linear Regression

# In[19]:


linear_regression = lnr()
pm_g = {}

lr = gsv(lnr(), pm_g, cv=6, scoring='neg_mean_squared_error', n_jobs=-2)
lr.fit(X_train, y_train)

finest_linear = lr.best_estimator_

y_p_li = finest_linear.predict(X_test)

m_li = me_err(y_test, y_p_li)
ma_li = ma_err(y_test, y_p_li)
r2_li = r2_accuracy(y_test, y_p_li)

print("linear regression metric:")
print("mean_squared_error:", m_li)
print("mean_absolute_error:", ma_li)
print("r_squared:", r2_li)
print()


# ## Gradient Boosting

# In[20]:


grad_boosting = grd()  
pm_g = { 'n_estimators': [100, 200, 300],'learning_rate': [0.05, 0.2, 0.3],'max_depth': [4, 5, 6]}

gr = gsv(grad_boosting, pm_g, cv=6, scoring='neg_mean_squared_error', n_jobs=-2) 
gr.fit(X_train, y_train)

finest_gradient = gr.best_estimator_ 

y_pr_g = finest_gradient.predict(X_test)
m_gr = me_err(y_test, y_pr_g)
ma_gr = ma_err(y_test, y_pr_g)
r2_gr = r2_accuracy(y_test, y_pr_g)

print("gradient boosting metric:")
print("mean_squared_error:", m_gr)
print("mean_absolute_error:", ma_gr)
print("r_squared:", r2_gr)
print()


# ## Random Forest

# In[21]:


random_forest = randf()

pm_g= {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20],'min_samples_split': [1, 4, 10],'min_samples_leaf': [2, 5, 6]}
rand_se = gsv(random_forest, pm_g, cv=6, scoring='neg_mean_squared_error', n_jobs=-2)  
rand_se.fit(X_train, y_train)

finest_random = rand_se.best_estimator_ 

y_p_r = finest_random.predict(X_test)
m_r = me_err(y_test, y_p_r)
ma_r= ma_err(y_test, y_p_r)
r2_r = r2_accuracy(y_test, y_p_r)

print("random forest metrics:")
print("mean_squared_error:", m_r)
print("mean_absolute_error:", ma_r)
print("r_squared:", r2_r)
print()


# ## Support Vector Machine

# In[22]:


svm_algorithm = sm()
pm_g = {'kernel': ['linear', 'rbf'],'C': [0.2, 2, 11],'gamma': ['scale', 'auto']}

svm_se = gsv(svm_algorithm, pm_g, cv=6, scoring='neg_mean_squared_error', n_jobs=-2)
svm_se.fit(X_train, y_train)

finest_svm = svm_se.best_estimator_

y_p_s = finest_svm.predict(X_test)

m_s = me_err(y_test, y_p_s)
ma_s = ma_err(y_test, y_p_s)
r2_s = r2_accuracy(y_test, y_p_s)

print("support vector machine metrics:")
print("mean_squared_error:", m_s)
print("mean_absolute_error:", ma_s)
print("r_squared:", r2_s)
print()


# ## K-Nearest Neighbors

# In[30]:


knn_algorithm = knr()  
pm_g = {'n_neighbors': [4, 6, 8, 10],'weights': ['uniform', 'distance']}

knn_se = gsv(knn_algorithm, pm_g, cv=6, scoring='neg_mean_squared_error', n_jobs=-2)  
knn_se.fit(X_train, y_train)

finest_knn = knn_se.best_estimator_ 

y_p_k = finest_knn.predict(X_test)
m_k = me_err(y_test, y_p_k)
ma_k = ma_err(y_test, y_p_k)
r2_k = r2_accuracy(y_test, y_p_k)

print("K-Nearest Neighbors Metrics:")
print("mean_squared_error:", m_k)
print("mean_absolute_error:", ma_k)
print("r_squared:", r2_k)


# #### These sections provide a detailed breakdown of each algorithms implementation, including hyperparameter tuning using grid search, model fitting, prediction, and evaluation metrics calculation.In which we have found Mean Squared Error,Mean Absolute Error and R-squared .

# # Best Model

# In[31]:


metrics_dict = {'Linear Regression': {'MSE': m_li, 'MAE': ma_li, 'R-squared': r2_li},'Gradient Boosting': {'MSE': m_gr, 'MAE':ma_gr, 'R-squared':r2_gr}, 'Random Forest': {'MSE':m_r, 'MAE': ma_r, 'R-squared':r2_r},'Support Vector Machine': {'MSE': m_s, 'MAE': ma_s, 'R-squared': r2_s},'K-Nearest Neighbors': {'MSE': m_k, 'MAE': ma_k, 'R-squared': r2_k}}

m_li = me_err(y_test, y_p_li)
ma_li = ma_err(y_test, y_p_li)
r2_li = r2_accuracy(y_test, y_p_li)

best_mse = min(metrics_dict, key=lambda x: metrics_dict[x]['MSE'])
best_mae = min(metrics_dict, key=lambda x: metrics_dict[x]['MAE'])
best_r2 = max(metrics_dict, key=lambda x: metrics_dict[x]['R-squared'])

print("best model based on MSE:", best_mse)
print("best model based on MAE:", best_mae)
print("best model based on R-squared:", best_r2)


# #### In this we have used computation matrics and find the model with lowest MSE then model with lowest MSE and in the end model with highest R Square.

# # Best model on R Square

# In[32]:


metrics_dict = {'Linear Regression': r2_li,'Gradient Boosting': r2_gr,'Random Forest': r2_r,'Support Vector Machine': r2_s,'K-Nearest Neighbors': r2_k}
finest_r2_model = max(metrics_dict, key=metrics_dict.get)

print("Best model based on R-squared:", finest_r2_model)
print("R-squared:", metrics_dict[finest_r2_model])


# #### Here we can see that Gradient Boosting is the best model based on R Square.

# # Predicting top 5 best stocks

# In[33]:


predictions = []
for target_variable in main_variables:
    y =nif50[target_variable]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gb_model = grd()
    gb_model.fit(X_train, y_train)
    
    future_predict = gb_model.predict(X_test)  
    
    predictions.append((target_variable, future_predict.mean()))

top_5_stocks = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

print("Top 5 stocks to invest in:")
for rank, (stock, _) in enumerate(top_5_stocks, 1):
    print(f"{rank}. {stock}")


# #### Here we have used Gradient Boosting for each target variable and based on our dataset we have predicted best 5 stocks to invest.

# # Conclusion

# ### Therefore by drawing these conclusions from above analysis, we have provided valuable insights to investors and stakeholders of company, helping them to make informed decisions and optimize their investment strategies for maximum returns in the Nifty 50 index.
