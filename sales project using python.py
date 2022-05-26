#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[35]:


#DATA_MANIBULATING
import pandas as pd
import numpy as np
#DATA-VISUALISATION
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # IMPORT DATASET

# In[36]:


df_sales=pd.read_csv('5000 Sales Records.csv')
df_sales


# In[3]:


#first 10 rows of dataset
df_sales.head(10)


# In[4]:


#last 10 rows of the dataset
df_sales.tail(10)


# In[5]:


#shape of the dataset
df_sales.shape


# In[6]:


#a concise summery of the dataset
df_sales.info()


# In[7]:


#getting descriptive statistics summary
df_sales.describe()


# In[8]:


#columns present in the dataset
df_sales.columns


# In[9]:


#chicking missing value
df_sales.isna().sum()


# # EXPLORATORY DATA ANALYSIS

# # WHICH IS THE TOP 10 COUNTRY BY TOTAL REVENUE  ?

# In[10]:


# Grouping country by revenue
country_revenue =pd.DataFrame(df_sales.groupby('Country').sum()['Total Revenue'])

# Sorting the dataframe in descending order
country_revenue.sort_values(by=['Total Revenue'], inplace=True, ascending=False)

# Top 10 countries by revenue
country_revenue[:10]


# # WHICH IS THE TOP 10 ITEM TYPE BY TOTAL PROFIT ?

# In[11]:


# Grouping item type by profit
item_profit =pd.DataFrame(df_sales.groupby('Item Type').sum()['Total Profit'])

# Sorting the dataframe in descending order
item_profit.sort_values(by=['Total Profit'], inplace=True, ascending=False)

# Top 10 item types by sales
item_profit[:10]


# # WHICH IS THE TOP 10 ITEM TYPE BY TOTAL COST ?

# In[12]:


# Grouping item type by cost
item_cost =pd.DataFrame(df_sales.groupby('Item Type').sum()['Total Cost'])

# Sorting the dataframe in descending order
item_cost.sort_values(by=['Total Cost'], inplace=True, ascending=False)

# Top 10 items types by cost
item_cost[:10]


# # WHAT IS THE OVERALL PROFIT TRENDS ?

# In[39]:


# change data type of "Order Date" and "Ship Date" to datetime format
df_sales['Order Date'] = pd.to_datetime(df_sales['Order Date'], format="%m/%d/%Y")
df_sales['Ship Date'] = pd.to_datetime(df_sales['Ship Date'], format="%m/%d/%Y")
df_sales.info()


# In[13]:


#convert type of order date column
df_sales['Order Date'] = pd.to_datetime(df_sales['Order Date'])


# In[14]:


# Getting month year from order date
df_sales['month_year'] = df_sales['Order Date'].apply(lambda x: x.strftime('%Y-%m'))


# In[15]:


# grouping month_year by sales
df_temp = df_sales.groupby('month_year').sum()['Total Profit'].reset_index()


# In[16]:


# Setting the figure size
plt.figure(figsize=(16, 5))
plt.plot(df_temp['month_year'], df_temp['Total Profit'], color='#b80045')
plt.xticks(rotation='vertical', size=7)
plt.show()


# # the columns that are categorical

# In[17]:


# columns that have less than 20 unique values
cat_cols_dic = {}
for i in range(len(df_sales.nunique())):
    if df_sales.nunique().values[i]<=20:
        cat_cols_dic[df_sales.nunique().index[i]] = df_sales.nunique().values[i]

print(cat_cols_dic)


# In[18]:


# region count visualization
plt.figure(figsize=[8,5])
sns.countplot(df_sales['Region'])
plt.title("Region Count", fontsize=14)
plt.xticks(rotation='vertical', size=14)
plt.show()


# In[19]:


# pie chat for item type
plt.figure(figsize=[6, 6])
plt.pie(df_sales['Item Type'].value_counts(normalize=True), labels= df_sales['Item Type'].value_counts().index, autopct="%.3f")
plt.title("Item Type counts", fontsize=14)
plt.show()


# In[20]:


# pie chat for sales channel
plt.figure(figsize=[6, 6])
plt.pie(df_sales['Sales Channel'].value_counts(normalize=True), labels= df_sales['Sales Channel'].value_counts().index, autopct="%.3f")
plt.title("Sales Channel counts", fontsize=14)
plt.show()


# In[21]:


# order priority count visualization
plt.figure(figsize=[8,5])
sns.countplot(df_sales['Order Priority'])
plt.title("Order Priority Count", fontsize=14)
plt.xticks(rotation='vertical', size=14)
plt.show()


# In[22]:


# pie chat for unit price
plt.figure(figsize=[6, 6])
plt.pie(df_sales['Unit Price'].value_counts(normalize=True), labels= df_sales['Unit Price'].value_counts().index, autopct="%.3f")
plt.title("Unit Price counts", fontsize=14)
plt.show()


# In[73]:


# unit cost count visualization
plt.figure(figsize=[16,5])
sns.countplot(df_sales['Unit Cost'])
plt.title("Unit Cost Count", fontsize=14)
plt.xticks(rotation='vertical', size=14)
plt.show()


# In[23]:


# Grouping products by Sales Channel and Item Type
cat_subcat = pd.DataFrame(df_sales.groupby(['Sales Channel', 'Item Type']).sum()['Total Profit'])

# Sorting the values
cat_subcat.sort_values(['Sales Channel','Total Profit'], ascending=False)


# # TIME SERIES ANALYSIS

# In[40]:


# First dates and last dates of Order dates and Ship dates
print("Describe Order date:")
print(df_sales['Order Date'].describe(), "\n")
print("Describe Ship date:")
print(df_sales['Ship Date'].describe())


# In[24]:


df_sales.set_index('Order Date',inplace=True)


# In[30]:


df_sales.plot(subplots=True, figsize=(20, 10));


# In[27]:


main_df = df_sales.sort_values(by='Order Date', ascending=True)[['Order Date','Total Revenue', 'Total Cost', 'Total Profit']]
main_df.set_index('Order Date', inplace=True)
main_df.head(10)


# In[42]:


# plot the sales graph
plt.figure(figsize=[15, 4])
plt.plot(df_sales, label='Profit')
plt.title('Sales(2015-2018)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Sales', fontsize=14)
plt.legend()
plt.show()


# In[16]:


#convert type of Ship Date column to datetime
df_sales['Ship Date'] = pd.to_datetime(df_sales['Ship Date'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




