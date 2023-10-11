#!/usr/bin/env python
# coding: utf-8

# The company is interested in getting to know:
# * What factors are important in estimating demand for shared bikes?
# * How effectively do those factors describe the demands of the bike?
# * The solution to this problem can be divided into the components that are listed below:
#     -  Understanding and performing data exploration
#     -  Data Visualisation
#     -  Data Preparation
#     -  Model building and evaluation

# # Step 1: Understanding and exploring data 

# In[1]:


import warnings
warnings.filterwarnings('ignore')             # Filtering/ignoring warnings


# In[2]:


# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


bike_df = pd.read_csv(r'C:\Users\LENOVO\Downloads\Bike_Sharing_Assignment\bike_data.csv')   #Loading dataset


# In[4]:


bike_df.shape     # display no.s of rows and columns in the dataset (730-rows, 16-columns)


# In[5]:


bike_df.info()    # displays summary of the dataset


# In[6]:


bike_df.describe()       # finding insights


# In[7]:


bike_df.isnull().sum() # checking for the null values in column data  (note: this dataset do not contain any null values)


# In[8]:


bike_df.dtypes  # learning datatypes in this dataset


# In[9]:


bike_df.nunique()       # finding unique values


# In[ ]:


# dropping the unwanted columns
bike_df.drop(['instant','dteday','casual','registered'],axis=1,inplace=True)


# In[53]:


bike_df.season = bike_df.season.map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})    # Encoding/mapping the season column


# In[55]:


# Encoding the month column
bike_df.month = bike_df.mnth.map({1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'june',7:'july',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'})


# In[56]:


# Mapping the weekday column
bike_df.weekday = bike_df.weekday.map({0:'sun',1:'mon',2:'tue',3:'wed',4:'thu',5:'fri',6:'sat'})


# In[57]:


# Encoding the weathersit column
bike_df.weathersit = bike_df.weathersit.map({1:'Clear',2:'Misty',3:'Light_snowrain',4:'Heavy_snowrain'})


# * So far, our understanding on data (Outcome) :
#   - Except for one column, the date, which is of the object type, the others are of the float or integer type. 
#   - Some fields are categorical in nature yet are in the integer/float type. 
#   - We have to analyse them and decide whether to convert them to categorical or treat them as integers.

# # Step 2: Data Visualisation (graphical representation of our data)

#  - Now, we'll move ahead and start performing one of the crucial steps: understanding the data.
#    - If there seems to be multicollinearity, this would be the first step where we can confirm it.
#    - This is also where we'll find out if some predictors strongly correlated  with the outcome variable.

# In[61]:


sns.pairplot(bike_df)      # Plot pairwise relationships in a dataset
plt.show()


# In[62]:


sns.distplot(bike_df['temp'])            # for univariant set of observations and visualisations (Temperature)
plt.show()


# In[63]:


sns.boxplot(bike_df['temp'])    # box-and-whisker plot shows the distribution of quantitative data
plt.show()


# In[64]:


sns.distplot(bike_df['hum'])      # distribution plot represents data in histogram form (Humidity)
plt.show()


# In[65]:


sns.boxplot(bike_df['hum'])    # visual representation of the depicting groups of numerical data through quartiles
plt.show()


# In[66]:


sns.distplot(bike_df['cnt'])   # cnt: count of total rental bikes including both casual and registered
plt.show()


# In[67]:


sns.boxplot(bike_df['cnt']) # distribution of quantitative data facilitates comparisons between variables of a categorical variable
plt.show()


# In[68]:


# Analysing the categorical columns to determine the way predictor variable stands against the target variable
plt.figure(figsize=(20, 12))
plt.subplot(2,4,1)
sns.boxplot(x = 'season', y = 'cnt', data = bike_df)
plt.subplot(2,4,2)
sns.boxplot(x = 'mnth', y = 'cnt', data = bike_df)
plt.subplot(2,4,3)
sns.boxplot(x = 'weekday', y = 'cnt', data = bike_df)
plt.subplot(2,4,4)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bike_df)
plt.subplot(2,4,5)
sns.boxplot(x = 'holiday', y = 'cnt', data = bike_df)
plt.subplot(2,4,6)
sns.boxplot(x = 'workingday', y = 'cnt', data = bike_df)
plt.subplot(2,4,7)
sns.boxplot(x = 'yr', y = 'cnt', data = bike_df)
plt.show()


# In[69]:


transformed = np.log10(bike_df['windspeed'])
sns.distplot(transformed)
print(transformed.skew())     # displays the direction of outliers


# In[70]:


transformed = np.log10(bike_df['windspeed'])**0.5
sns.distplot(transformed)             # represents the overall distribution of continuous variables
print(transformed.skew())


# In[71]:


from sklearn.preprocessing import PowerTransformer     # provides several common utility functions to change raw feature vectors into  suitable downstream estimators.
pwer_trns = PowerTransformer(method='box-cox')                # to make data more Gaussian-like
tranformed = pwer_trns.fit_transform(bike_df[['windspeed']])
sns.distplot(tranformed)
print(transformed.skew())


# In[72]:


bike_df['transformedwindspeed'] = pwer_trns.fit_transform(bike_df[['windspeed']])


# In[102]:


bike_df_numeric = bike_df.select_dtypes(include=['float64'])   # All numeric variables in the dataset
bike_df_numeric.head()


# In[103]:


sns.pairplot(bike_df_numeric)    # Pairwise Scatter Plot
plt.show()


# In[104]:


cor = bike_df_numeric.corr()    #Correlation Matrix
cor


# In[105]:


mask= np.array(cor)           # Heatmap
mask[np.tril_indices_from(mask)]= False
fig,ax= plt.subplots()
fig.set_size_inches(10,10)
sns.heatmap(cor,mask=mask,vmax=.8,square=True,annot=True)
plt.show()


# In[106]:


bike_df.drop('atemp',axis=1,inplace=True)  #removing atemp as it is highly correlated with temp


# ## Step 3: Data Preparation

# In[73]:


# Creating dummy variable for month, weekday, weathersit and season variables.
mnths_df=pd.get_dummies(bike_df.mnth,drop_first=True)
weekdays_df=pd.get_dummies(bike_df.weekday,drop_first=True)
weathersit_df=pd.get_dummies(bike_df.weathersit,drop_first=True)
seasons_df=pd.get_dummies(bike_df.season,drop_first=True)


# In[74]:


bike_df.head()


# In[107]:


bike_df_categorical=bike_df.select_dtypes(include=['object'])   #Subset of all categorical variables


# Dummy Variables

# Variable e.g. season,mnth,weekday and weathersit have different levels, and now need to convert these levels into integers.

# In[108]:


bike_df_dummies = pd.get_dummies(bike_df_categorical, drop_first=True)   # Convert into dummies
bike_df_dummies.head()


# In[109]:


bike_df = bike_df.drop(list(bike_df_categorical.columns), axis=1)


# In[110]:


bike_df = pd.concat([bike_df,bike_df_dummies],axis=1)


# In[111]:


bike_df.head()


# In[75]:


# Merging  the dataframe, with the dummy variable dataset
bike_df_new = pd.concat([bike_df,mnths_df,weekdays_df,weathersit_df,seasons_df],axis=1)


# In[76]:


bike_df_new.head()


# In[77]:


bike_df_new.info()


# In[78]:


bike_df_new.drop(['season','mnth','weekday','weathersit'], axis = 1, inplace = True) # # dropping unnecessary columns 


# In[79]:


bike_df_new.head()    # displays the head of new dataframe


# In[80]:


bike_df_new.shape     # displays the shape of new dataframe


# In[81]:


bike_df_new.info()    # displays the column info of new dataframe


# # Step 4: Splitting the data into training & testing sets

# In[116]:


# importing libraries to train our model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import r2_score


# In[117]:


np.random.seed(0)                                  # splitting the data into Train and Test
datafrm_train, datafrm_test = train_test_split(bike_df_new, train_size = 0.7, random_state = 100)


# In[118]:


datafrm_train.shape        # displays the shape of training datatset


# In[119]:


datafrm_test.shape         # displays the shape of testing datatset


# In[120]:


scaler = MinMaxScaler()    # Using MinMaxScaler to rescale features


# In[121]:


datafrm_train.head()      # viewing the head of dataset before scaling


# In[122]:


num_vars = ['temp','atemp','hum','windspeed','cnt'] # Applying scaler() to all colmns except'yes-no' & 'dummy' variables
datafrm_train[num_vars] = scaler.fit_transform(datafrm_train[num_vars])


# In[123]:


datafrm_train.head()


# In[124]:


datafrm_train.describe()   # describing the dataset


# In[125]:


plt.figure(figsize = (25,25))      # displays correlation coefficients to see which variables are highly correlated
matrix = np.triu(datafrm_train.corr())
sns.heatmap(datafrm_train.corr(), annot = True, cmap="RdYlGn", mask=matrix)
plt.show()


# Here, cnt appears to have correlation with year (yr) and temperature (tem) variables. Also, misty and humidity have a similar correlation. The spring season correlates well with January and February months, the summer season with the month of May, and the winter season with the months of October and November.

# In[126]:


plt.figure(figsize=[6,6])                # Visualising one of the correlation to find trends with scatter plot
plt.scatter(datafrm_train.temp, datafrm_train.cnt)
plt.show()


# Therefore, our visualization techniques confirms the positive correlation between temp and cnt

# Model building and evaluation

# In[134]:


#Split the dataframe into train and test sets
from sklearn.model_selection import train_test_split
np.random.seed(0)
datafrm_train, datafrm_test = train_test_split(bike_df, train_size=0.7, test_size=0.3, random_state=100)


# In[135]:


datafrm_train


# In[ ]:




