# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:34:12 2022

@author: Diana
"""
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2



from pandas import read_csv

from pydoc import help
from scipy.stats.stats import pearsonr
help(pearsonr)


#import date############
diab=read_csv('diabet.csv')
print(diab)
diab.info()

diab.shape
diab.info()

print(diab.describe())


from pandas import set_option
set_option('display.width', 100)
set_option('precision', 3)
description = diab.describe()
print(description)

diab.describe().T
diab.isnull().sum()



a=1
plt.figure(figsize=(20, 10))
for i in diab.columns:
    plt.subplot(3, 3, a)
    sns.distplot(diab[i])
    a += 1
plt.show()

diab.BloodPressure.tail()


diab.loc[diab['BloodPressure']<80, 'Blood_category'] = 'normal pressure'
diab.loc[diab['BloodPressure']>=80, 'Blood_category'] = 'high p.stg 1'
diab.loc[diab['BloodPressure']>=90 , 'Blood_category'] = 'high p.stg 2'
diab.loc[diab['BloodPressure']>120 , 'Blood_category'] = 'hipertensive crisis'
print(diab.Blood_category)

print(diab.Blood_category)
print(diab)

plt.hist(diab['Blood_category'])
plt.title("BloodPressure")
plt.show()

bmi_category = ["underweight", "normal", "overweight", "obesity_1", "obesity_2", "obesity_3"]
diab['BMI_Cat'] = pd.cut(diab['BMI'], [-1, 18.5, 25, 30, 35, 40, diab['BMI'].max()], 
                       labels=bmi_category)

Freq_bmi_category=diab['BMI_Cat'].value_counts()
print(Freq_bmi_category)
plot=Freq_bmi_category.plot.pie()
plt.show()

print(diab)

diab.describe()
print(diab.BMI_Cat.value_counts())
print(diab.Blood_category.value_counts())

diab.Glucose.hist()

diab.hist(figsize=(15,7.5))
plt.show()






diab.plot(kind='box',figsize=(15,10),subplots=True,layout=(3,3))
plt.show() 


def Z_Score_outlier_handling(column):
    upper_limit = diab[column].mean() + 3*diab[column].std()
    lower_limit = diab[column].mean() - 3*diab[column].std()
    diab[column] = np.where(
        diab[column] > upper_limit, upper_limit,
        np.where(diab[column] < lower_limit, lower_limit, 
        diab[column])
    )

# Outlier treatment by column
Z_Score_outlier_handling("Glucose")
Z_Score_outlier_handling("BloodPressure")
Z_Score_outlier_handling("SkinThickness")
Z_Score_outlier_handling("BMI")


diab.plot(kind='box',figsize=(15,10),subplots=True,layout=(3,3))
plt.show() 

def IQR_outlier_handling(column):
    percentile25 = diab[column].quantile(0.25)
    percentile75 = diab[column].quantile(0.75)
    iqr = percentile75-percentile25

    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    diab[diab[column] > upper_limit]
    diab[diab[column] < lower_limit]
    
    new_df = diab[diab[column] < upper_limit]
    new_df.shape
    
    diab[column] = np.where(
    diab[column] > upper_limit,
    upper_limit,
    np.where(
        diab[column] < lower_limit,
        lower_limit,
        diab[column]
        )
    )
# Outlier treatment by column

IQR_outlier_handling("Insulin")
IQR_outlier_handling("DiabetesPedigreeFunction")
IQR_outlier_handling("Age")


diab.plot(kind='box',figsize=(15,10),subplots=True,layout=(3,3))
plt.show() 


#analiza de asociere
db=pd.crosstab(diab.BMI_Cat,
               diab.Blood_category,
               margins=False)
print(db)
rezultat_chisquare= chi2_contingency(db)
print(rezultat_chisquare)

Freq_bmi_category=np.array(diab.BMI_Cat)
import numpy as np

db=pd.crosstab(diab.BMI_Cat,diab.Blood_category,margins=True)
print(db)

db=pd.crosstab(diab.Blood_category,diab.BMI_Cat,margins=True)
print(db)

db=pd.crosstab(diab.BMI_Cat,diab.Blood_category,margins=False)
print(db)
import scipy as sp
import scipy.stats
import statsmodels.stats.api as sms
print('confidence interval',sms.DescrStatsW(diab.BloodPressure).tconfint_mean())

from scipy import stats
print(stats.ttest_1samp(diab.BMI, 0))

bmi_low=diab.loc[diab['BMI_Cat'] == 'underweight']
bmi_high=diab.loc[diab['BMI_Cat'] == 'overweight']
print(stats.ttest_ind(bmi_low.Glucose, bmi_high.Glucose))
print(stats.ttest_ind(bmi_low.Glucose, bmi_high.Glucose, equal_var=False))





import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Glucose~BMI_Cat', data=diab).fit()
print(sm.stats.anova_lm(model, typ=2))

#Ho: Nu exista diferenta intre nivelul de glucoză si greutatea corporala
#H1: Exista diferenta intre nivelul de glucoză si greutatea corporala

Insulin=diab.Insulin
X=Insulin
X=sm.add_constant(X)
model=sm.OLS(diab.Glucose, X)
results=model.fit()
print(results.summary())

import statsmodels.api as sm
BMI=diab.BMI
X=BMI
X=sm.add_constant(X)
model=sm.OLS(diab.Glucose, X)
results=model.fit()
print(results.summary())

#regresie multipla
x_multiple=pd.DataFrame({'Insulin':diab.Insulin,'BloodPressure':diab['BloodPressure'],'BMI':diab['BMI']})
x_multiple=sm.add_constant(x_multiple)
y=diab.Glucose
model_multiple=sm.OLS(y,x_multiple)
results_multiple=model_multiple.fit()
print(results_multiple.summary())


plt.figure(figsize=(12,10))
cor = diab.corr()
sns.heatmap(cor, annot=True,cmap=plt.cm.Reds)
plt.show()



#the positive correlation 
print('Insulin->SkinThickness')
print(pearsonr(diab.Insulin,diab.SkinThickness))
print('BMI->SkinThickness')
print(pearsonr(diab.BMI,diab.SkinThickness))
print('Insulin->Glucose')
print(pearsonr(diab.Insulin,diab.Glucose))
print('BMI->BloodPressure')
print(pearsonr(diab.BMI,diab.BloodPressure))
print('Age->Glucose')
print(pearsonr(diab.Age,diab.Glucose))
print('Age->BloodPressure')
print(pearsonr(diab.Age,diab.BloodPressure))
 #the negative correlation
print('Age->SkinThickness')
print(pearsonr(diab.Age,diab.SkinThickness))
print('Age->Insulin')
print(pearsonr(diab.Age,diab.Insulin))


#regresie multipla
x_multiple=pd.DataFrame({'Insulin':diab.Insulin,'BloodPressure':diab['BloodPressure'],'BMI':diab['BMI']})
x_multiple=sm.add_constant(x_multiple)
y=diab.Glucose
model_multiple=sm.OLS(y,x_multiple)
results_multiple=model_multiple.fit()
print(results_multiple.summary())

#the positive correlation 
print('Insulin->SkinThickness')
print(pearsonr(diab.Insulin,diab.SkinThickness))
print('BMI->SkinThickness')
print(pearsonr(diab.BMI,diab.SkinThickness))
print('Insulin->Glucose')
print(pearsonr(diab.Insulin,diab.Glucose))
print('BMI->BloodPressure')
print(pearsonr(diab.BMI,diab.BloodPressure))
print('Age->Glucose')
print(pearsonr(diab.Age,diab.Glucose))
print('Age->BloodPressure')
print(pearsonr(diab.Age,diab.BloodPressure))
 #the negative correlation
print('Age->SkinThickness')
print(pearsonr(diab.Age,diab.SkinThickness))
print('Age->Insulin')
print(pearsonr(diab.Age,diab.Insulin))

print('Age->bmi')
print(pearsonr(diab.Age,diab.BMI))


X_multiple=pd.DataFrame({'BMI':diab.BMI,'BMI2':diab.BMI**2})
X_multiple=sm.add_constant(X_multiple)
Y=diab.Glucose 
model_nelin_patratic=sm.OLS(Y,X_multiple)
result_nelin_patratic=model_nelin_patratic.fit()
print(result_nelin_patratic.summary())


X_multiple=pd.DataFrame({'Insulin':diab.Insulin,'Insulin2':diab.Insulin**2})
X_multiple=sm.add_constant(X_multiple)
Y=diab.Glucose 
model_nelin_patratic=sm.OLS(Y,X_multiple)
result_nelin_patratic=model_nelin_patratic.fit()
print(result_nelin_patratic.summary())

print('Model liniar multiplu==', results_multiple.rsquared)
print('Model neliniar patratic==', result_nelin_patratic.rsquared)





