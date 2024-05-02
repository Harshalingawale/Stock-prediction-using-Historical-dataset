#!/usr/bin/env python
# coding: utf-8

# # Project Topic

# ## Getting Investment Insights By Analyzing Nifty 50 Data.

# # Problem Statement  
# Link to Dataset --[click here](https://www.kaggle.com/datasets/setseries/nifty50-stocks-dataset20102021/data?select=Final-50-stocks.csv)

# ### A investment consulting firm has a dataset of Nifty 50 stocks and they want to gain actionable insights from this dataset. After gaining insights they want to invest and guide there clients to buid a profitable portfolio. Hence they have hired me as Data Scientist to Analyze the dataset in detail and come up with some potential insights.

# # Import Libraries

# In[1]:


import pandas as pd
import seaborn as abc
import datetime as dt
import numpy as np
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[6]:


nif50 = pd.read_csv("datasets/Final-50-stocks.csv")


# #### loading the dataset 

# # Data Exploration 

# In[7]:


nif50.info()


# In[8]:


nif50.dtypes


# In[9]:


nif50.columns


# # Data Processing and Cleaning

# In[10]:


nif50['DATE']=pd.to_datetime(nif50.DATE)
nif50['Year']=nif50.DATE.dt.year 
nif50['Month']=nif50.DATE.dt.month


# #### In this we converted the values in the date, column of the dataframe nif50 into datetime objects. Then we create a new column named month, year in the dataframe nif50.

# In[11]:


nif50.HEROMOTOCO.isnull().value_counts()


# #### This applies a boolean mask to the column 'HEROMOTOCO', returning a boolean Series where each element is True if the corresponding value in the column is NaN (null), and False otherwise.

# In[12]:


nif50.HEROMOTOCO.fillna(nif50.HEROMOTOCO.mean(),inplace=True)
nif50.HEROMOTOCO


# In[13]:


nif50.COALINDIA.isnull().value_counts()
nif50.COALINDIA.fillna(nif50.COALINDIA.mean(),inplace=True)
nif50.COALINDIA


# In[14]:


nif50.UPL.isnull().value_counts()
nif50.UPL.fillna(nif50.UPL.mean(),inplace=True)
nif50.UPL


# In[15]:


nif50.ADANIPORTS.isnull().value_counts()
nif50.ADANIPORTS.fillna(nif50.ADANIPORTS.mean(),inplace=True)
nif50.ADANIPORTS


# In[16]:


nif50.INFY.isnull().value_counts()
nif50.INFY.fillna(nif50.ADANIPORTS.mean(),inplace=True)
nif50.INFY


# #### This replaces missing (NaN) values with a specified value and calculates the mean value of the non-missing values and after that the parameter specifies that the changes should be applied directly to nif50.

# In[17]:


clean_data = nif50.isna().sum().sum()
print("Null Values in dataset - ", clean_data)


# #### Therefore all Null Values are removed.

# # Data Visualization 
# 1.Displot for data distribution for 10 stocks

# In[18]:


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


# # Data Analysis 

# ## 1.what is the performance of Stocks from IT, BANKING And PHARMA Sector ?

# ### 1. IT SECTOR

# In[19]:


plot.figure(figsize=(15, 6))
abc.set_style('whitegrid')
plot.title('Tech stocks over the Years')

abc.lineplot(x=nif50['Year'], y=nif50['TCS'], label='TCS')
abc.lineplot(x=nif50['Year'], y=nif50['HCLTECH'], label='HCLTECH')
abc.lineplot(x=nif50['Year'], y=nif50['WIPRO'], label='WIPRO')
abc.lineplot(x=nif50['Year'], y=nif50['TECHM'], label='TECH MAHINDRA')

plot.xlabel('Years')
plot.ylabel('Stock Price')
plot.legend()

plot.show()


# ### 2. BANKING SECTOR

# In[20]:


plot.figure(figsize=(15, 6))
abc.set_style('whitegrid')
plot.title('Banks over the years')

abc.lineplot(x=nif50['Year'], y=nif50['HDFCBANK'], label='HDFC', color='blue')
abc.lineplot(x=nif50['Year'], y=nif50['SBIN'], label="SBI", color='yellow')
abc.lineplot(x=nif50['Year'], y=nif50['KOTAKBANK'], label='KOTAK MAHINDRA', color='green')
abc.lineplot(x=nif50['Year'], y=nif50['ICICIBANK'], label='ICICI', color='red')
abc.lineplot(x=nif50['Year'], y=nif50['AXISBANK'], label='AXISBANK', color='gray')

plot.xlabel('Years')
plot.ylabel('Stock Price')
plot.legend()

plot.show()


# ### 3. Pharma Sector

# In[21]:


plot.figure(figsize=(15, 6))
abc.set_style('whitegrid')
plot.title('Pharma stocks')

abc.lineplot(x=nif50['Year'], y=nif50['SUNPHARMA'], color='red', label='Sun Pharma')
abc.lineplot(x=nif50['Year'], y=nif50['DRREDDYS'], color='green', label="Dr. Reddy's")
abc.lineplot(x=nif50['Year'], y=nif50['CIPLA'], color='blue', label='Cipla')

plot.xlabel('Years')
plot.ylabel('Stock Price')

plot.legend()

plot.show()


# #### Here we have used lineplot to depict the movement of IT, BANKING and PHARMA Stocks from 2010-2021 in which TCS from IT Sector grew exponentially compared to its competitors . In this we can also see that there is a sudden change of momentum in TECH MAHINDRA at 2014 , where as in WIPRO we don't see any major change of momentum.For BANKING Sector we can see that SBI had a good start but later compared to other stocks did not perform well, whereas KOTAK MAHINDRA had a decent start is still performing good as compared to others. In PHARMA sector DR REDDY outperformed and continues to be unbeatable as compared to CIPLA and SUNPHARMA.

# ## 2. What is the Daily, Monthly And Yearly Rate of Return in Nifty 50 Stocks ?
# 

# In[23]:


average_daily_change=[]
average_monthly_change=[]
average_yearly_change=[]

d=[]
m=[]
y=[]

stock=list(nif50.columns)
for i in stock:
    if i=='DATE' or i=='Year' or i=='Month':
        continue
    for j in range(len(stock)-1):
        price_change=nif50[i][j+1]-nif50[i][j]
        rate_daily_change=(price_change/nif50[i][j])*100
        d.append(rate_daily_change)
    for k in range(0,len(stock)-1,30):
        price_change_monthly=nif50[i][k+30]-nif50[i][k]
        rate_monthly_change=(price_change_monthly/nif50[i][k])*100
        m.append(rate_monthly_change)
    for l in range(0,len(stock)-1,365):
        price_change_yearly=nif50[i][l+365]-nif50[i][l]
        rate_yearly_change=(price_change_yearly/nif50[i][l])*100
        y.append(rate_yearly_change)

new_data=pd.DataFrame()
new_data['Stocks']=pd.Series(stock)
new_data['Daily_Returns']=pd.Series(d)
new_data['Monthly_Returns']=pd.Series(m)
new_data['Years_Returns']=pd.Series(y)
new_data.dropna(inplace=True)
new_data.dropna(inplace=True)
new_data.drop([0],inplace=True)
new_data.Daily_Returns=round(new_data.Daily_Returns,2)
new_data.Monthly_Returns=round(new_data.Monthly_Returns,2)
new_data.Years_Returns=round(new_data.Years_Returns,2)


# ### Stocks with Highest Daily Returns
# 

# In[24]:


new_data.sort_values('Daily_Returns', ascending=False, inplace=True)

plot.figure(figsize=(16, 6))
abc.set_style('whitegrid')
plot.title('Stocks with highest daily returns')
abc.barplot(x=new_data['Stocks'].head(10), y=new_data['Daily_Returns'].head(10))
plot.xlabel('Stocks')
plot.ylabel('% Returns')

plot.show()


# ### Stocks with Highest Monthly Returns
# 

# In[25]:


new_data.sort_values('Monthly_Returns', ascending=False, inplace=True)

plot.figure(figsize=(15, 6))
abc.set_style('whitegrid')
plot.title('Stocks with highest Monthly returns')
abc.barplot(x=new_data['Stocks'].head(10), y=new_data['Monthly_Returns'].head(10))
plot.xlabel('Stocks')
plot.ylabel('% Returns')
plot.show()


# ### Stocks with Highest Yearly Returns

# In[26]:


new_data.sort_values('Years_Returns', ascending=False, inplace=True)

plot.figure(figsize=(15, 6))
abc.set_style('whitegrid')
plot.title('Stocks with highest returns')
abc.barplot(x=new_data['Stocks'].head(10), y=new_data['Years_Returns'].head(10))
plot.xlabel('Stocks')
plot.ylabel('% Returns')
plot.show()


# #### Here we have first sorted 10 stocks in ascending order who have gained highest return from the new_data dataset and used barplot to plot the data and displayed the stocks who gave highest daily, monthly and yearly return . In the stocks with daily return we can obeserve that IOC has given highest amount of return whereas TITAN significantly gave more return as compaired to other for monthly return and INDUSBANK outperformed and gave highest returns as to others.

# # 3.What will be the effect of Correlation between top 10 stock's daily closing price ?
# 

# In[27]:


top_10_stocks = nif50.mean().nlargest(10).index
top10_stocks_data = nif50[top_10_stocks]

plot.figure(figsize=(20, 10))
for stock in top_10_stocks:
    plot.plot(top10_stocks_data.index, top10_stocks_data[stock], label=stock)

plot.title('Daily Closing Prices of Top 10 Stocks')
plot.xlabel('Date')
plot.ylabel('Closing Price')
plot.legend(loc='upper left')
plot.xticks(rotation=45)
plot.grid(True)
plot.show()


# #### In the following question we have first created a dataset where top 10 stocks can be saved, and then we have used the original dataset called my data where mean value of top 10 stocks is calculated and to display this we have used plt plot to plot the data . In this we can obesrve that EICHERMOTOR had a good closing price but later failed to maintain its price whereas SHREECEMENT is the only one stock who has maintained its daily closing price.

# # 4.what stocks can the investors keep in there portfolio to have positive return?
# 

# In[28]:


new_data.describe().transpose()


# In[31]:


average_daily_rate_of_return=[]
average_monthly_rate_of_return=[]
average_yearly_rate_of_return=[]
l=[]
q=[]
r=[]
stocks=list(nif50.columns)
for col in stocks:
    if(col=='DATE'):
        continue
    for j in range(len(col)-1):
        price_change=nif50[col][j+1]-nif50[col][j]
        rate_of_change=(100*price_change)/nif50[col][j]
        l.append(rate_of_change)
    for j in range(0,len(col),30):
        price_change=nif50[col][j+30]-nif50[col][j]
        rate_of_change=(100*price_change)/nif50[col][j]
        q.append(rate_of_change)
    for j in range(0,len(col),365):
        price_change=nif50[col][j+365]-nif50[col][j]
        rate_of_change=(100*price_change)/nif50[col][j]
        r.append(rate_of_change)
    
    average_daily_rate_of_return.append(sum(l)/len(l))
    average_monthly_rate_of_return.append(sum(q)/len(q))
    average_yearly_rate_of_return.append(sum(r)/len(r))
    l=[]
    q=[]
    r=[]


new_data['Avg Daily Rate of Return'] = pd.Series(average_daily_rate_of_return)
new_data['Avg Monthly Rate of Return'] = pd.Series(average_monthly_rate_of_return)
new_data['Avg Yearly Rate of Return'] = pd.Series(average_yearly_rate_of_return)


# In[32]:


monthly_stocks=new_data[new_data['Avg Monthly Rate of Return']>5]
yearly_stocks=new_data[new_data['Avg Yearly Rate of Return']>10]

monthly_stocks[['Stocks','Avg Monthly Rate of Return']].plot(x='Stocks',kind='bar')
plot.xlabel('Stocks')
plot.title('Stocks giving positive monthly returns (>5%) ')
plot.show()
# In[33]:


yearly_stocks[['Stocks','Avg Yearly Rate of Return']].plot(x='Stocks',kind='bar')
plot.xlabel('Stocks')
plot.title('Stocks giving positive yearly returns (>10%) ')
plot.show()


# #### For the investors who are seeking to get positive return in there portfolio we have obtained the output of average monthly  rate of return and average yearly rate of return in which for monthly return we have displayed stocks who gave positive return more than 5 percent and for stocks who gave yearly positive return more than 10 percent which is displayed using a bar chart .

# # 5.What are the outliers in terms of daily trading volumes?
# 

# In[34]:


z_scores = (new_data['Avg Daily Rate of Return'] - new_data['Avg Daily Rate of Return'].mean()) / new_data['Avg Daily Rate of Return'].std()

outliers = new_data[abs(z_scores) > 3]

new_data['Outlier'] = new_data.index.isin(outliers.index)


# In[35]:


plot.figure(figsize=(10, 6))
plot.scatter(new_data.index, new_data['Avg Daily Rate of Return'], c=new_data['Outlier'], cmap='coolwarm', s=100)
plot.plot(new_data.index, new_data['Avg Daily Rate of Return'].rolling(window=10).mean(), color='green', label='Trend Line')

outlier_size = 100
non_outlier_size = 50
sizes = [outlier_size if x else non_outlier_size for x in new_data['Outlier']]

plot.legend(['Trend Line', 'Outlier', 'Non-Outlier'])

plot.xlabel('Company Index')
plot.ylabel('Avg Daily Rate of Return')
plot.title('Average Daily Rate of Return with Outliers')

plot.grid(True)
plot.show()


# #### Here firstly we have installed plotly and then defined outliers by simply Calculating z-scores for Avg Daily Rate of Return. then we have used scatter plot to calculate average daily rate of return with outliers highlighted and then Added a trend line for better visualization. In this we can observe that company index number 7 which is SHREECEM has outperformed as compaired to others.

# # 6.How do the returns of different stocks vary over time? 

# In[36]:


plot.figure(figsize=(12, 8))  

top_10_data = new_data[new_data['Stocks'].isin(top_10_stocks)]

abc.boxplot(data=top_10_data, x='Stocks', y='Daily_Returns', palette='Blues', showmeans=True, meanprops={"marker":"o",
                   "markerfacecolor":"white", 
                   "markeredgecolor":"black",
                   "markersize":"10"})

plot.title('Distribution of Daily Returns for Top 10 Stocks Over Time')
plot.xlabel('Stocks')
plot.ylabel('Daily Returns')
plot.xticks(rotation=45, ha='right')  
plot.grid(True)
plot.tight_layout()
plot.show()


# #### In the following question to get the returns of different stocks who vary over time we have Create a box plot for daily returns of each of the top 10 stocks. We have Shown mean and standard deviation in the box plot. In this we can see that BAJAJFINSERV gave a good return of max 2.75 upper fence and MARUTI standing in second position which max 2.29 upper fence.

# # 7.Are there any stocks that frequently experience price reversals after significant price movements ? 

# In[38]:


returns = nif50.drop(columns=['DATE']).pct_change()

sh_trm_ret = returns.rolling(window=5).mean().mean()
lo_trm_ret = returns.rolling(window=30).mean().mean()

top_20_sh_trm_ret = sh_trm_ret.nlargest(20)
top_20_lo_trm_ret = lo_trm_ret.nlargest(20)

always_reversals = {}

for column in returns.columns:
    positive_movements = returns[column][returns[column] > 0]
    negative_movements = returns[column][returns[column] < 0]

    always_reversals[column] = {
        'positive': len(positive_movements) - (positive_movements.shift(1) < 0).sum(),
        'negative': len(negative_movements) - (negative_movements.shift(1) > 0).sum()
    }

reversals_df = pd.DataFrame(always_reversals).T
top_10_stocks = reversals_df.sum(axis=1).nlargest(10).index

top_10_reversals_df = reversals_df.loc[top_10_stocks]

plot.figure(figsize=(12, 8))
top_10_reversals_df.plot(kind='bar', stacked=True)

plot.title('Top 10 Stocks - Frequency of Positive and Negative Reversals')
plot.xlabel('Stock')
plot.ylabel('Frequency')
plot.xticks(rotation=45, ha='right')
plot.legend(['Positive', 'Negative'], title='Reversal Type')
plot.grid(axis='y', linestyle='--', alpha=0.7)

plot.tight_layout()
plot.show()


# #### Here we calculated daily percentage changes in stock prices for each column then defined threshold for significant price movements and calculated it. later we Identified instances where the daily percentage change exceeds the threshold and define time frame for reversal detection and later we Identify instances of significant positive and negative movements and Check if subsequent price movement always reverses the trend within the specified time frame and finally  plot the chart.

# ## Conclusion

# ### Therefore by drawing these conclusions from above analysis, we have provided valuable insights to investors and stakeholders of company, helping them to make informed decisions and optimize their investment strategies for maximum returns in the Nifty 50 index.we can also observe that company such as TITAN, INDUSBANK, BAJAJFINSERV have a good potential overall to invest.
