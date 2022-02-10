import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df_c = pd.read_csv("C:\\Users\\Asus\\Downloads\\Cog =2\\Cleaned_Automobile.csv")
# print(df_c.isnull().sum())

print(f"Descriptive analysis of the Data Set : \n {df_c.describe()} \n ")
df_c.rename(columns={'make': 'Company'}, inplace=True)

Window1 = plt.figure(1)
sns.boxplot(x='Drive-wheels', y='price', data=df_c)
#plt.savefig("Box-plot of Type of Drive Wheels")
Window2 = plt.figure(2)
sns.boxplot(x='Body-style', y='price', data=df_c)
#plt.savefig("Box-plot of Different Body Styles of cars")
Window3 = plt.figure(3)
sns.boxplot(x='Company', y='price', data=df_c)
#plt.savefig("Box-plot of Companies or Manufactures of Cars")

#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
print(f"Data Type of Categories of the Data Set \n {df_c.dtypes} ")

#------------------------------------------------------------------------------------------
cor_col = df_c[['bore', 'stroke', 'Compression - ratio', 'horsepower']].corr()
print(f"Correlation between Categories - bore, stroke, Compression-ratio, horsepower \n {cor_col}")
Window4 = plt.figure(4)
plt.pcolor(cor_col, cmap='RdBu')
plt.colorbar()
plt.title("Heatmap of Correlation between Categories - bore, stroke, Compression-ratio, horsepower")
#plt.savefig("Correlation Representation by Heatmap")
#-----------------------------------------------------------------------------------------------------
# Relationship of continuous Numerical Variables -  Scatter -  regplot( with regression line)
#pearson coef - gives 2 values
# coef
# P-test or P-value
#in the form of tuple
Window5 = plt.figure(5)
sns.regplot(x='engine-size', y='price', data=df_c)
plt.xlabel("Engine-Style")
plt.ylabel("Price")
plt.title("Relationship between size of engines and Price")
cor_value = df_c[['engine-size', 'price']].corr()
pearson_cor_coef = stats.pearsonr(df_c['engine-size'], df_c['price'])
print(f"Correlation between size of engines and Price : \n{cor_value}\n "
      f"Pearson Correlation Coefficient : {pearson_cor_coef} \n ")
#plt.savefig("Correlation of Engine type and price using scatter plot")

#price and engine-size have strong positive linear correlation
#[elements affecting price  - 1) engine-size ... ] - These list of elements may affect the target

Window6 = plt.figure(6)
sns.regplot(x='horsepower', y='price', data=df_c)
plt.xlabel("horsepower")
plt.ylabel("Price")
plt.title("Relationship between Horsepower and Price")
cor_value_1 = df_c[['horsepower', 'price']].corr()
pearson_cor_coef_1 = stats.pearsonr(df_c['horsepower'], df_c['price'])
print(f"Correlation between Horsepower and Price : \n{cor_value_1}\n "
      f"Pearson Correlation Coefficient : {pearson_cor_coef_1} \n ")
#plt.savefig("Correlation of Horsepower and price using scatter plot")

#price and horsepower have strong positive linear correlation
#[elements affecting price  - 1) engine-size 2)horsepower]

Window7 = plt.figure(7)
sns.regplot(x='city-L/100km', y='price', data=df_c)
plt.xlabel("City-L/100km")
plt.ylabel("Price")
plt.title("Relationship between Litre per 100 km in cities and Price")
cor_value_2 = df_c[['city-L/100km', 'price']].corr()
pearson_cor_coef_2 = stats.pearsonr(df_c['city-L/100km'], df_c['price'])
print(f"Correlation between Litre per 100 km in cities and Price : \n{cor_value_2}\n "
      f"Pearson Correlation Coefficient : {pearson_cor_coef_2} \n ")
#plt.savefig("Correlation of Litre per 100 km in cities and price using scatter plot")

#price and city L/100km have Moderate to strong positive linear correlation
#[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km]

Window8 = plt.figure(8)
sns.regplot(x='highway-L/100km', y='price', data=df_c)
plt.xlabel("highway-L/100km")
plt.ylabel("Price")
plt.title("Relationship between Litre per 100 km on highways and Price")
cor_value_3 = df_c[['highway-L/100km', 'price']].corr()
pearson_cor_coef_3 = stats.pearsonr(df_c['highway-L/100km'], df_c['price'])
print(f"Correlation between Litre per 100 km on highways and Price : \n{cor_value_3}\n "
      f"Pearson Correlation Coefficient : {pearson_cor_coef_3} \n ")
#plt.savefig("Correlation of Litre per 100 km on highways and price using scatter plot")

#price and highway L/100km have  strong positive linear correlation
#[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km 4)highway-L/100km]

Window9 = plt.figure(9)
sns.regplot(x='peak-rpm', y='price', data=df_c)
plt.xlabel("peak-rpm")
plt.ylabel("Price")
plt.title("Relationship between rotation per minute and Price")
cor_value_4 = df_c[['peak-rpm', 'price']].corr()
pearson_cor_coef_4 = stats.pearsonr(df_c['peak-rpm'], df_c['price'])
print(f"Correlation between rotation per minute and Price : \n{cor_value_4}\n "
      f"Pearson Correlation Coefficient : {pearson_cor_coef_4} \n ")
#plt.savefig("Correlation of Rotation per minute and price using scatter plot")

#price and peak-rpm has very weak negative linear relationship - we can say close to no relationship
#[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km 4)highway-L/100km]

Window10 = plt.figure(10)
sns.regplot(x='stroke', y='price', data=df_c)
plt.xlabel("stroke")
plt.ylabel("Price")
plt.title("Relationship between strokes and Price")
#plt.savefig("Correlation of Stroke and price using scatter plot")
plt.show(block=False)
plt.close(fig='all')
cor_value_5 = df_c[['stroke', 'price']].corr()
pearson_cor_coef_5 = stats.pearsonr(df_c['stroke'], df_c['price'])
print(f"Correlation between strokes and Price : \n{cor_value_5}\n "
      f"Pearson Correlation Coefficient : {pearson_cor_coef_5} \n ")

#price and stroke has very weak positive linear relationship - we can say close to no relationship
#[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km 4)highway-L/100km]

#------------------------------------------------------------------------------------------------------------------------------
#Categorical
# in window2 - the boxes are overlapping so would not have positive impact on data

Window11 = plt.figure(11)
sns.boxplot(x='Engine-location', y='price', data=df_c)
plt.show(block=False)
plt.close('all')
#here boxes are distinguish and ca have positive impact on target
#[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km 4)highway-L/100km 5)Engine-location]

# Window 1 can have positive impact on target
#[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km 4)highway-L/100km
# 5)Engine-location 6)drive-wheels]

#--------------------------------------------------------------------------------------------------------------------------------
#Descriptive

temp = df_c[['Drive-wheels', 'Body-style']].value_counts().to_frame()
temp.rename(columns={0: 'Value-Count'}, inplace=True)
print(temp.columns)
print(temp)

temp_1 = df_c[['Drive-wheels', 'Body-style', 'Company']].value_counts().to_frame()
temp_1.rename(columns={0: 'Value-Count'}, inplace=True)
print(temp_1.columns)
print(temp_1)

Unique_Count_Drive = df_c['Drive-wheels'].value_counts().to_frame()
Unique_Count_Drive.rename(columns={'Drive-wheels': 'Value-Counts'}, inplace=True)
Unique_Count_Drive.index.name = 'Drive-wheels'
print(f"\n\nSummary of Types of Drive Wheels \n {Unique_Count_Drive}\n")


Unique_Count_Engine_loc = df_c['Engine-location'].value_counts().to_frame()
Unique_Count_Engine_loc.rename(columns={'Engine-location': 'Value-Counts'}, inplace=True)
Unique_Count_Engine_loc.index.name = 'Engine-location'
print(f"\n\nSummary of Location of Engine \n {Unique_Count_Engine_loc}\n")

Unique_Count_HB = df_c['horsepower-binned'].value_counts().to_frame()
Unique_Count_HB.rename(columns={'horsepower-binned': 'Value-Counts'}, inplace=True)
Unique_Count_HB.index.name = 'horsepower-binned'
print(f"\n\nSummary of Location of Engine \n {Unique_Count_HB}\n")

#------------------------------------------------------------------------------------------------------------------
#Grouping - split - apply- merge
df_grp1 = df_c[['Drive-wheels', 'price']]
df_grp1_avg = df_grp1.groupby(['Drive-wheels'], as_index=False).mean()
df_grp1_avg.set_index('Drive-wheels', inplace=True)
df_grp1_avg.rename(columns={'price': 'avg-price'}, inplace=True)
df_grp1_avg.sort_values(ascending=False, by='avg-price', inplace=True)
print(f"Most Valuable Type of Drive wheel on the basis of average \n {df_grp1_avg}\n ")

df_grp2 = df_c[['Drive-wheels', 'Engine-location', 'price']]
df_grp2_avg = df_grp2.groupby(['Drive-wheels', 'Engine-location'], as_index=False).mean()
df_grp2_avg.set_index('Drive-wheels', inplace=True)
df_grp2_avg.rename(columns={'price': 'avg-price'}, inplace=True)
df_grp2_avg.sort_values(ascending=False, by='avg-price', inplace=True)
print(f"Most Valuable Type of Drive wheel and engine location on the basis of average \n {df_grp2_avg}\n ")

df_grp2_pivot = pd.pivot_table(df_grp2_avg, index='Drive-wheels', columns='Engine-location', values='avg-price')
df_grp2_pivot.fillna(0, inplace=True)
print(df_grp2_pivot)

df_grp3 = df_c[['Drive-wheels', 'Body-style', 'price']]
df_grp3_avg = df_grp3.groupby(['Drive-wheels', 'Body-style'], as_index=False).mean()
df_grp3_avg.set_index('Drive-wheels', inplace=True)
df_grp3_avg.rename(columns={'price': 'avg-price'}, inplace=True)
df_grp3_avg.sort_values(ascending=False, by='avg-price', inplace=True)
print(f"Most Valuable Type of Drive wheel and engine location on the basis of average \n {df_grp3_avg}\n ")

df_grp3_pivot = pd.pivot_table(df_grp3_avg, index='Drive-wheels', columns='Body-style', values='avg-price')
df_grp3_pivot.fillna(0, inplace=True)
print(df_grp3_pivot)

#--------- Rel using Heatmap ------------------------
Window12 = plt.figure(12)
plt.subplot(1, 2, 1)
sns.heatmap(df_grp3_pivot, cmap='RdBu', xticklabels=True, yticklabels=True)
plt.subplot(1, 2, 2)
sns.heatmap(df_grp2_pivot, cmap='Blues', xticklabels=True, yticklabels=True)
#plt.savefig('2 heatmaps showing relationship cat1 and cat2 vs price')
plt.show(block=False)
plt.close('all')

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#observing length, width and height
Window13 = plt.figure(13)
plt.subplot(3, 1, 1)
sns.regplot(x='length', y='price', data=df_c)
plt.xlabel('length')
plt.ylabel('Price')
plt.title("Relationship between Length and price")
plt.subplot(3, 1, 2)
sns.regplot(x='width', y='price', data=df_c)
plt.xlabel('width')
plt.ylabel('Price')
plt.title("Relationship between width and price")
plt.subplot(3, 1, 3)
sns.regplot(x='height', y='price', data=df_c)
plt.xlabel('height')
plt.ylabel('Price')
plt.title("Relationship between height and price")
plt.show(block=False)
plt.close('all')

le = stats.pearsonr(df_c['length'], df_c['price'])
wid = stats.pearsonr(df_c['width'], df_c['price'])
hei = stats.pearsonr(df_c['height'], df_c['price'])
print(f"Length \n Pearson's coefficient = {le[0]} with P-test value = {le[1]}")
print(f"\nWidth \n Pearson's coefficient = {wid[0]} with P-test value = {wid[1]}")
print(f"\nHeight \n Pearson's coefficient = {hei[0]} with P-test value = {hei[1]}")
#from this data we can see length and width makes moderately positive linear relation while
#height makes very weak positive relation, near to 0
#so length and width may be useful in predicting the target

##[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km 4)highway-L/100km
# 5)Engine-location 6)drive-wheels 7)length 8)width]

Window14 = plt.figure(14)
plt.subplot(2, 1, 1)
sns.regplot(x='Wheel-base', y='price', data=df_c)
plt.xlabel('Wheel-base')
plt.ylabel('price')
plt.subplot(2, 1, 2)
sns.regplot(x='bore', y='price', data=df_c)
plt.xlabel('bore')
plt.ylabel('price')
plt.show(block=False)
plt.close('all')

Wb = stats.pearsonr(df_c['Wheel-base'], df_c['price'])
bor = stats.pearsonr(df_c['bore'], df_c['price'])
print(f"Wheel Base \n Pearson's coefficient = {Wb[0]} with P-test value = {Wb[1]}")
print(f"bore \n Pearson's coefficient = {bor[0]} with P-test value = {bor[1]}")
#Wheel base and Bore shows Moderate positive linear relation with price, hence they can be predictor for price

##[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km 4)highway-L/100km
# 5)Engine-location 6)drive-wheels 7)length 8)width 9)Wheel-Base 10)bore]

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#ANOVA
##[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km 4)highway-L/100km
# 5)Engine-location 6)drive-wheels 7)length 8)width 9)Wheel-Base 10)bore]
#as drive-wheels is the most strongly related among categorical data, the anova test should give us
# High value of F-Test Score and smol p-value

df_grp_temp_anv = df_c[['Drive-wheels', 'price']].groupby(['Drive-wheels'])
avg = df_grp_temp_anv.mean()
avg.rename(columns={'price': 'avg-price'}, inplace=True)
avg.reset_index(inplace=True)

Window15 = plt.figure(15)
sns.barplot(x='Drive-wheels', y='price', data=df_c)
plt.xlabel('Drive-wheels')
plt.ylabel('price')

Window16 = plt.figure(16)
sns.barplot(x='Drive-wheels', y='avg-price', data=avg)
plt.xlabel('Drive-wheels')
plt.ylabel('Average price')
plt.show(block=False)
plt.close('all')

#4wd vs rwd
anova1 = stats.f_oneway(df_grp_temp_anv.get_group('4wd')['price'], df_grp_temp_anv.get_group('rwd')['price'])

#rwd vs fwd
anova2 = stats.f_oneway(df_grp_temp_anv.get_group('rwd')['price'], df_grp_temp_anv.get_group('fwd')['price'])

#fwd vs 4wd
anova3 = stats.f_oneway(df_grp_temp_anv.get_group('fwd')['price'], df_grp_temp_anv.get_group('4wd')['price'])

print(f" 4wd vs rwd \n F-Test Value - {anova1[0]} and p value - {anova1[1]} \n "
      f" rwd vs fwd \n F-Test Value - {anova2[0]} and p value - {anova2[1]} \n "
      f" fwd vs 4wd \n F-Test Value - {anova3[0]} and p value - {anova3[1]}")

#------------------------------------------------------------------------------------------------------------------------

#Conclusion - Features affecting Target

##[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km 4)highway-L/100km
# 5)Engine-location 6)drive-wheels 7)length 8)width 9)Wheel-Base 10)bore]

# Non categorical Data
# - Engine-size, horsepower, city-L/100km, highway-l/100km, length, width, Wheel-Base, bore

# Categorical Data
# - Engine-location, Drive-wheels

#------------------------------------------------------------------------------------------------------------------------








