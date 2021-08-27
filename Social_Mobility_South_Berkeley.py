#!/usr/bin/env python
# coding: utf-8

# # Inferential Statistics Assignment
# ## ~ Upward Mobility in South Berkeley ~ 
# 
# ### Date: July/2021
# 
# ### Author: Kazutoki Matsui
# ##### (Discussed with Vrashabh, Jose, Rita, and Sam at a high level)

# ## Part I

# ### Q1: Upward Mobility: All Income and Race
# <br>
# <font size="3">
#     
#   In this assignment, I will look at South Berkeley area, since I am an international student. 
#     
#   Looking at the Atlas map, South Berkeley have around average mobility for Black and White people, but less than the state and national average for Hispenic population. Also I noticed that Hispanic people have higher rate of incarceration at 6.3% than the national average.</font>
# <h3><center>(Exhibit1) Tract 06001424001, South Berkeley, Berkeley, CA<br>Low Income Households, All Race and Gender</h3>
# 
# ![image.png](attachment:image.png)
#     
# ### <h3><center>(Exhibit2) Low Income Households, Hispanic and pooled_Gender<br>Hispanic people have significantly lower income than the national average.</h3>
# ![image.png](attachment:image.png)
#     

# In[1]:


## import several packages
import pandas as pd
import numpy as np
from statsmodels import api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Read the dataset and look at the first few observations
df = pd.read_stata('atlas.dta')
df.head()


# In[3]:


## Check the columns names
df.columns


# ### Q2: Hypothesis on the Mobility in Berkeley
# <font size="3">White and Asian people in Bay Area are generally upper class households. Race could matter. Also. household environment such as single-parent could affect the future income, since the households have lower chance of sending their children to college. Or even in worse cases, the children might commit a crime and get incarcerated, which could lead to umemployment in the future.
#     
# So my hypothesis to analyze is:
# 1. Race of their parents, especially non-white families
# 2. Single parent households
#     
# matter when predicting the income in the adulthood.
# </font>
# </font>

# ### Q3: Average upward mobility estimates
# <font size="3">The variable ['kfr_pooled_pooled_p25'] measures the average income rank in adulthood for children who grew up in a given tract and whose parents are at the 25th percentile of the national income distribution. 
# 
# As we look at the summary statistics below, the varialbe can go negative and over 1.00. This is because they are just estimates, and it is estimated using a sample observations and not the entire population. Since the estimate is based on a sample, a statistical model is used to estimate the statistic.
#     
# The researchers added "independent, normally distributed noise to each of these estimates" to maintain privacy of the source individuals. The authors take care of this issues because some tracts have very few sample observations and the privacy of the individuals should be protected when constructing publicly available statistics. As a result, depending on the standard errors, the estimate does not necessarily add up to 100%, and sometimes could include values outside 0 to 1.</font>

# In[4]:


## Check the summary statistic of 'kfr_pooled_pooled_p25'
df['kfr_pooled_pooled_p25'].describe()


# In[21]:


# Since State and County codes are recognized as numeric, I omitted their first zeros in each factor. 
'''Parameters'''
tract_number = '06001424001'
State = 'CA'
County = 'Berkeley'
Tract = 'South Berkeley'
state = 6
county = 1
tract = 424001
'''Parameters'''
    
tr = df[(df['state']==state) & (df['county']==county) & (df['tract']==tract)]
tr


# ### Q4: Average upward mobility [kfr_pooled_pooled_p25] in Fillmore, compared to CA state and US.

# <font size="3">The average of mobility in fillmore is lower than state and national average, meaning that overall chance of "climing" the ladder is lower than other regions. </font>

# In[23]:


## 'kfr_pooled_pooled_p25' in Fillmore
tr_average = tr['kfr_pooled_pooled_p25'].item()
## County Average
ct_average = df[(df['state']==state) & (df['county']==county)]['kfr_pooled_pooled_p25'].mean()
## State Average of 'kfr_pooled_pooled_p25'
st_average = df[df['state']==state]['kfr_pooled_pooled_p25'].mean()
## US Average
us_average = df['kfr_pooled_pooled_p25'].mean()

print(f'The average upward of mobility in {Tract} is {tr_average}')
print(f'The average upward of mobility in {County} is {ct_average}')
print(f'The average upward of mobility in {State} is {st_average}')
print(f'The average upward of mobility in US is {us_average}')


# ### Q5 Standard Deviation of Upward Mobility, ['kfr_pooled_pooled_p25']

# <font size="3">Berkeley has higher variation in the mobility than CA state average, but it is around the national average.</font>

# In[7]:


tr_std = df[(df['state']==state) & (df['county']==county)]['kfr_pooled_pooled_p25'].std()
st_std = df[df['state']==state]['kfr_pooled_pooled_p25'].std()
us_std = df['kfr_pooled_pooled_p25'].std()

print(f'The Standard Deviation of upward of mobility in {County} is {tr_std}')
print(f'The Standard Deviation of upward of mobility in {State} is {st_std}')
print(f'The Standard Deviation of upward of mobility in US is {us_std}')


# ### Q6 Histram of 'kfr_pooled_pooled_p25' in County 

# <font size="3">The distribution is centered around roughly the mean 0.35, but it has smaller tail on the right hand side, meaning that it is right-skewed.<font>

# In[8]:


plt.figure(figsize=(10, 5))
X = df[(df['state']==state) & (df['county']==county)]['kfr_pooled_pooled_p25']
plt.hist(X, bins=40) 
plt.title(f"Histogram of Upward Mobility in {County}")
plt.xlabel('Upward Mobility')
plt.ylabel('Count')
plt.show()


# ### Q7: Average Mobility by Race

# <font size="3">The comparison of the average between tract, state, and national shows that South Berkeley have around average mobility for Black and White people, but less than the state and national average for Hispenic population. 
# 
# On the other hand, the standard deviations of mobility in South Berkeley are higher for White population, but for Black and Hispanic people, they are roughly the same as the national average.</font>

# In[25]:


## Tract Average by Race
tr_baverage = tr['kfr_black_pooled_p25'].item()
tr_haverage = tr['kfr_hisp_pooled_p25'].item()
tr_waverage = tr['kfr_white_pooled_p25'].item()
## County Average by Race
ct_baverage = df[(df['state']==state) & (df['county']==county)]['kfr_black_pooled_p25'].mean()
ct_haverage = df[(df['state']==state) & (df['county']==county)]['kfr_hisp_pooled_p25'].mean()
ct_waverage = df[(df['state']==state) & (df['county']==county)]['kfr_white_pooled_p25'].mean()
## State Average by Race
st_baverage = df[df['state']==state]['kfr_black_pooled_p25'].mean()
st_haverage = df[df['state']==state]['kfr_hisp_pooled_p25'].mean()
st_waverage = df[df['state']==state]['kfr_white_pooled_p25'].mean()
## US Average by Race
us_baverage = df['kfr_black_pooled_p25'].mean()
us_haverage = df['kfr_hisp_pooled_p25'].mean()
us_waverage = df['kfr_white_pooled_p25'].mean()

print(f" ## {Tract} Average by Race")
print(f'The average upward of mobility for Black people in {Tract} is {tr_baverage}')
print(f'The average upward of mobility for Hispanic people in {Tract} is {tr_haverage}')
print(f'The average upward of mobility for White people in {Tract} is {tr_waverage}')
print(f"\n ## {County} County Average by Race")
print(f'The average upward of mobility for Black people in {County} is {ct_baverage}')
print(f'The average upward of mobility for Hispanic people in {County} is {ct_haverage}')
print(f'The average upward of mobility for White people in {County} is {ct_waverage}')
print(f"\n ## {State} State Average by Race")
print(f'The average upward of mobility for Black people in {State} is {st_baverage}')
print(f'The average upward of mobility for Hispanic people in{State} is {st_haverage}')
print(f'The average upward of mobility for White people in {State} is {st_waverage}')
print("\n ## US Average by Race")
print(f'The average upward of mobility for Black peoplein US is {us_baverage}')
print(f'The average upward of mobility for Hispanic people in US is {us_haverage}')
print(f'The average upward of mobility for White people in US is {us_waverage}')


# ### Standard Deviation by Race

# <font size="3">The comparison of mobility in volatility shows that Black and Hispanic population are around the national average, while white people have higher mobility than the nation as a whole.

# In[10]:


## County Std by Race
tr_bstd = df[(df['state']==state) & (df['county']==county)]['kfr_black_pooled_p25'].std()
tr_hstd = df[(df['state']==state) & (df['county']==county)]['kfr_hisp_pooled_p25'].std()
tr_wstd = df[(df['state']==state) & (df['county']==county)]['kfr_white_pooled_p25'].std()

## State Std by Race
st_bstd = df[df['state']==state]['kfr_black_pooled_p25'].std()
st_hstd = df[df['state']==state]['kfr_hisp_pooled_p25'].std()
st_wstd = df[df['state']==state]['kfr_white_pooled_p25'].std()
## US Std by Race
us_bstd = df['kfr_black_pooled_p25'].std()
us_hstd = df['kfr_hisp_pooled_p25'].std()
us_wstd = df['kfr_white_pooled_p25'].std()

print(f" ## {Tract} Standard Deviation by Race")
print(f'The Standard Deviation of upward of mobility for Black people in {County} is {tr_bstd}')
print(f'The Standard Deviation of upward of mobility for Hispanic people in {County} is {tr_hstd}')
print(f'The Standard Deviation of upward of mobility for White people in {County} is {tr_wstd}')
print(f"\n ## {State} State Standard Deviation by Race")
print(f'The Standard Deviation of upward of mobility for Black people in {State} is {st_bstd}')
print(f'The Standard Deviation of upward of mobility for Hispanic people in {State} is {st_hstd}')
print(f'The Standard Deviation of upward of mobility for White people in {State} is {st_wstd}')
print("\n ## US Standard Deviation by Race")
print(f'The Standard Deviation of upward of mobility for Black peoplein US is {us_bstd}')
print(f'The Standard Deviation of upward of mobility for Hispanic people in US is {us_hstd}')
print(f'The Standard Deviation of upward of mobility for White people in US is {us_wstd}')


# ### Histogram by Race

# In[11]:


x = df[(df['state']==state) & (df['county']==county)]
fig, axs = plt.subplots(3,figsize=(15,10))
axs[0].set_title('Black Households')
axs[0].set_xlim([0.2,0.8])
axs[1].set_title('Hispanic Households')
axs[1].set_xlim([0.2,0.8])
axs[2].set_title('White Households')
axs[2].set_xlim([0.2,0.8])
axs[0].hist(x['kfr_black_pooled_p25'], bins=40, color = "skyblue")
axs[1].hist(x['kfr_hisp_pooled_p25'], bins=40, color = "limegreen")
axs[2].hist(x['kfr_white_pooled_p25'], bins=40, color = "indianred")
axs[2].set(ylabel='count')
axs[2].set(xlabel='Upward Mobility')
fig.tight_layout()

plt.show()


# ### Q8 Correlation Matrix of Mobility in County 

# <font size="3"> I checked with all the combination of the variables with 'kfr_pooled_pooled_p25', and sort them in ascending order. Considering that the target population is 30s in their life, I assumed we should focus on 2000, when they are growing up as a child and are affected by the surrounding environments.

# In[12]:


## To see the correlation of the variables, take a look at the correlation matrix
df_county = df[(df['state']==state) & (df['county']==county)]
columns =['kfr_pooled_pooled_p25','hhinc_mean2000',
       'mean_commutetime2000', 'frac_coll_plus2010', 'frac_coll_plus2000',
       'foreign_share2010', 'med_hhinc2016', 'med_hhinc1990', 'popdensity2000',
       'poor_share2010', 'poor_share2000', 'poor_share1990', 'share_black2010',
       'share_hisp2010', 'share_asian2010', 'share_black2000',
       'share_white2000', 'share_hisp2000', 'share_asian2000',
        'rent_twobed2015', 'singleparent_share2010',
       'singleparent_share1990', 'singleparent_share2000', 'traveltime15_2010',
       'emp2000', 'mail_return_rate2010', 'ln_wage_growth_hs_grad',
       'jobs_total_5mi_2015', 'jobs_highpay_5mi_2015', 'nonwhite_share2010',
       'popdensity2010', 'ann_avg_job_growth_2004_2013', 'job_density_2013',
        'jail_pooled_pooled_p25',
       'kfr_black_pooled_p25', 'kfr_hisp_pooled_p25', 'kfr_white_pooled_p25',
       'jail_black_pooled_p25', 'jail_hisp_pooled_p25',
       'jail_white_pooled_p25', 'kfr_pooled_female_p25', 'kfr_pooled_male_p25',
       'jail_pooled_female_p25', 'jail_pooled_male_p25',
       'kfr_black_female_p25', 'kfr_hisp_female_p25', 'kfr_white_female_p25',
       'kfr_black_male_p25', 'kfr_hisp_male_p25', 'kfr_white_male_p25',
       'jail_black_female_p25', 'jail_hisp_female_p25',
       'jail_white_female_p25', 'jail_black_male_p25', 'jail_hisp_male_p25',
       'jail_white_male_p25']
df_county = df_county[columns]
df_county.corr(method= 'pearson').sort_values('kfr_pooled_pooled_p25')
## Look at the first column to see the correlations with 'kfr_pooled_pooled_p25'. Correlations are sorted by ascending order.


# In[15]:


columns_r =['kfr_pooled_pooled_p25','singleparent_share2000','share_asian2000', 'share_hisp2000','share_black2000']
df_county_r = df_county[columns_r]
df_county_r.corr(method= 'pearson').sort_values('kfr_pooled_pooled_p25', ascending = False)


# ### Q9 Scatter matrix of some of the variables

# <font size="3">Created the scatter matrix below, ['kfr_pooled_pooled_p25','singleparent_share2000', 'singleparent_share1990','share_black2010','poor_share1990', 'poor_share2000', 'share_asian2010','foreign_share2010','mean_commutetime2000'] </font>

# In[16]:


import seaborn as sns

sns.pairplot(df_county_r)
plt.show()
## Look at the first column to see the correlations with 'kfr_pooled_pooled_p25'. Correlations are sorted by descending order.


# ### Q10 Non-Linear Relationship
# <font size="3">All the variables ['singleparent_share2000','share_asian2000', 'nonwhite_share2010'] seems to have some non-linear relationship from the scatter plots above. I took the log of X variables and regressed against 'kfr_pooled_pooled_p25' and got them all statistically significant, meaning that they indeed have non-linear relationships.</font>

# In[17]:


## take log of X variables
df_county = df[(df['state']==state) & (df['county']==county)]
df_county = df_county[columns]
df_county_log = df_county
df_county_log[columns_r[1:]] = df_county_log[columns_r[1:]].apply(lambda x: np.log(x), axis= 1)
df_county_log[columns_r].head()


# In[18]:


## Y = a + b1*ln(X1) + b2*ln(X2) + b3*ln(X3)
model = smf.ols(formula= 'kfr_pooled_pooled_p25 ~ singleparent_share2000 + share_asian2000 +share_hisp2000 + share_black2000', data=df_county_log)
results_log = model.fit()

print(results_log.summary())


# ## Part II

# ### Q11: Hypothesis Testing
# 

# <font size="3">All the variables being born in a black neiborhood does matter for the future income, reducing the income rank by 2%. While hispanic variable is also significant, the effect seems very small, meaning share of hispanic people does not predict the income in adulthood. On the other hand, born in asian neighborhoold implies positive relationship with future income.
# The R squard is around 70%, which I assume it is relatively well explained.</font>

# In[19]:


## Run regression with a few explanatory variables
# df_county = df_county.dropna()
model = smf.ols(formula= 'kfr_pooled_pooled_p25 ~ singleparent_share2000 + share_asian2000 +share_hisp2000+share_black2000', data=df_county)
results_lin = model.fit()

print(results_lin.summary())


# ### Q12: Narrative Interpretation

# <font size="3">I have tried a bunch of combinations of variables to run the regressions, while narrowing the patters based on some common sense (e.g. poor_1990 and poor_2000 should incur too high correlation, so I did not include them at once). Unexpectedly, Basically, most of the varialbes were NOT statistically significant, so I did not include the results here. Whereas Variables such as 'poor_share1990','med_hhinc1990' have highly correlated with the mobility variable, they are NOT significant. This is somewhat counterintuitive way since the economic environment of the household should be a big factor where the child grows up.
# 
# As my conclusion, mainly TWO factors matter when predicting a child future income (i.e. mobility).
# 1. If a child is born in a sigle-parent family, this reduces the rank of the future income of the child by 4.6 percentile.
# 2. Race matters:
#   <br> Black: if a child is born in a black neighborhoods, this reduces the rank of the future income of the child by 1.8 percentile.
#   <br>Hispanic: being hispanic significantly decreases the rank in national scale. This is contrary to Fillmore district in San Francisco where hispanic people earned significantly more than national average. 
#   <br>White: n Berkeley, white population earns higher than other races, but they have more variation in the future income, implying more mobility.
#    <br>Asian: Asian neighborhood have positive correlation with the future income of children. This implies that these Asians could be immgrants or children of immigrants, and they are earning good amount of income. The areas they live tend to have higher rents and have better environment to raise their children.
#     
# This trend above makes sense when looking at the histograms in Part I, showing that the distributions are apprently different by race (especially black households seem to be trapped in lower income). Again, the comparison in Q7 which shows the average between tract, state, and national shows that South Berkeley have around average mobility for Black and White people, but less than the state and national average for Hispanic population.</font>
