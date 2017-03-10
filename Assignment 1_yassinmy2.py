
# coding: utf-8

# # ASSIGNMENT - BY YASSIN MY

# In[245]:

import pandas as pd
import numpy as np


# In[246]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns; 
from sklearn.linear_model import LinearRegression
from scipy import stats
import pylab as pl

sns.set()


# In[247]:

telco_churn = pd.read_csv(r'C:\Users\Yassin\Desktop\test\Pandas-Exercises-master\Telco-Customer-Churn.csv')


# In[248]:

telco_churn.head() #always show first 5 as set as default


# #### Info about the dataset?

# In[249]:

telco_churn.info()


# In[250]:

telco_churn.shape #total columns and rows


# #### What is the name of 20th column?¶

# In[251]:

telco_churn.columns[19]


# #### How is the dataset indexed?

# In[252]:

telco_churn.index


# #### Count female and male customer

# In[253]:

pd.crosstab(index=telco_churn["gender"],columns="Count") # count male and female customer
                      


# ### RELATIONSHIP BETWEEN SENIOR CITIZEN AND CHURN STATUS ###

# #### Count senior citizen who being the customer for the telco company

# In[254]:

telco_churn.groupby('SeniorCitizen').size()   # count no of senior citizen customer


# In[255]:

IDSeniorCitizen = telco_churn[['customerID', 'SeniorCitizen', 'Churn']] # show per ID
IDSeniorCitizen


# In[256]:

IDSeniorCitizen.groupby(['Churn', 'SeniorCitizen']).size()


#         #### Based on the figure above, 476 senior citizen has churned whereas the other 666 has retained the services.

# In[257]:

PerChurnSrCitizen = IDSeniorCitizen.groupby(['Churn','SeniorCitizen']).Churn.count()
PerChurnSrCitizen_per_ps = IDSeniorCitizen.groupby(['SeniorCitizen']).SeniorCitizen.count()

PerChurnSrCitizen/PerChurnSrCitizen_per_ps*100


#         #### The percentage obtained above shows 41.68% senior citizen has churned whereas the other 58.32% has retained the services.

# In[258]:

telco_churn.groupby('Contract').size()


#        #### Based on the figure above, month-to-month description has the highest customers, followed by two years, and one year packages.

# #### Visualizing the obtained finding

# In[259]:

telco_churn.hist(column="SeniorCitizen",by="Churn",bins=30)


# ### RELATIONSHIP BETWEEN PACKAGE SUBSCRIPTION AND CHURN STATUS

# In[260]:

telco_churn.groupby('Contract').size()


# In[261]:

telco_churn['Contract'].unique()
contract_group = telco_churn.groupby('Contract').apply(lambda x: len(x))
contract_group

contract_group.plot(kind='barh', grid=True)
plt.ylabel('Contract Subscription')
plt.xlabel('No of Customer')
plt.title('No of Customer for Contract Subscription')


#         #### Month-to-month subscription shows the highest customers subscribed.

# In[262]:

telco_churn.groupby(['Churn', 'Contract']).size()


#         #### Out of 3875 customers who subscribed month-to-month package, 1655 has churned from the telco company.

# ### Visualization of the obtained data above. 

# In[263]:


def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded
 


# In[264]:

print 'Before Nominalize:' 
print pd.value_counts(telco_churn["Contract"])
telco_churn["Contract_Nominal"] = coding(telco_churn["Contract"], {'Month-to-month':0,'One year':1,'Two year':2}) #replace churn value Yes=1 and No=0:
print '\nAfter Nominalize:'
print pd.value_counts(telco_churn["Contract_Nominal"])


# In[265]:

telco_churn.hist(column="Contract_Nominal", by="Churn")
plt.xlabel('Contract Subscription')
plt.ylabel('No of Customer')


# ### RELATIONSHIP BETWEEN NO OF SENIOR CITIZEN, PACKAGE SUBSCRIPTION AND CHURN STATUS

# In[266]:

telco_churn.groupby(['SeniorCitizen', 'Contract']).size()


#         #### Out of 3875 customers who subscribed month-to-month package, 807 are senior citizens. The data illustrated in hist below.

# In[267]:

telco_churn.hist(column="Contract_Nominal", by="SeniorCitizen")
plt.xlabel('Contract Subscription')
plt.ylabel('No of Customer')
plt.title('Senior Citizen')


# In[268]:

telco_churn.groupby(['SeniorCitizen', 'Contract','Churn' ]).size()


#         #### From the data above, 441 senior citizens who subscribed the month-to-month package has churned.

# ### What is the relationship between these 3 attributes?

# In[269]:

PerChurnSrCitizen_per_package = telco_churn.groupby(['SeniorCitizen','Contract','Churn']).SeniorCitizen.count()
PerChurnSrCitizen_per_ps2 = telco_churn.groupby(['SeniorCitizen']).SeniorCitizen.count()

Data = PerChurnSrCitizen_per_package/PerChurnSrCitizen_per_ps*100
Data


#     #### Above percentage shows that 38.62% of senior citizen who subscribed the month-to-month pacakge has churned which indicates the highest population among the senior citizens. It can be shown in diagram below.

# In[270]:

Data.plot(kind='barh', grid=True)


# ## LINEAR REGRESSION ##

# In[271]:

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14


# In[285]:

X = telco_churn.MonthlyCharges[0:30]
Y = telco_churn.Contract_Nominal[0:30] # plot loike hahaha!!
plt.ylabel('Contract')
plt.xlabel('Total Charge')
plt.title('Scatter Plot of Tenure by Total Charges')
plt.plot(X, Y, 'o');


#         #### Linear regression is not suitable for this project since attributes chosed are suit.

# In[303]:

# Pandas scatter plot
telco_churn.plot(kind='scatter', x='Contract_Nominal', y='MonthlyCharges', alpha=0.2)


#     ### Based on the graph plotted above, Linear Regression model is not suitable to analyze the chosen attributes.

# ## Spliting data into test and train data set

# In[273]:

X = telco_churn.ix[:, telco_churn.columns != 'Churn']
Y = telco_churn['Churn']

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)


# In[274]:

X_test.info()


# ### Appendix

# #### Nominalize the value of Churn as Yes = 1, and No = 0

# In[275]:


def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded
 

print 'Before Nominalize:' 
print pd.value_counts(telco_churn["Churn"])
telco_churn["Churn_Nominal"] = coding(telco_churn["Churn"], {'No':0,'Yes':1}) #replace churn value Yes=1 and No=0:
print '\nAfter Nominalize:'
print pd.value_counts(telco_churn["Churn_Nominal"])

