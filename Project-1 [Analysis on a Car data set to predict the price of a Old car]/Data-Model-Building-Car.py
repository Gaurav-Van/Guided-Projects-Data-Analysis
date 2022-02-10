import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score  #predict value -  scoring - stores in array ....  default scoring is r^2
from sklearn.model_selection import cross_val_predict  #result before r^2 score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score  #for polynomial regression model #otherwise lm.score(will work)


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


def PollyPlot(xtrain, xtest, ytrain, ytest, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    # training data
    # testing data
    # lr:  linear regression object
    # poly_transform:  polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])

    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, ytrain, 'ro', label='Training Data')
    plt.plot(xtest, ytest, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()


df_c = pd.read_csv("C:\\Users\\Asus\\Downloads\\Cog =2\\Cleaned_Automobile.csv")

y_data = df_c['price']
x_data = df_c.drop('price', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
print(f"number of test samples : {x_test.shape[0]}")
print(f'number of training samples: {x_train.shape[0]} ')

# ---------------------------------------------------------------------------------------------------------------------------------
##[elements affecting price  - 1) engine-size 2)horsepower 3)city-L/100km 4)highway-L/100km
# 5)Engine-location 6)drive-wheels 7)length 8)width 9)Wheel-Base 10)bore]

# Non categorical Data
# - Engine-size, horsepower, city-L/100km, highway-l/100km, length, width, Wheel-Base, bore

# Categorical Data
# - Engine-location, Drive-wheels
# --------------------------------------------------------------------------------------------------------------------------------===

# Simple Linear Reg - ax + b
# Multiple Linear - ax2 + a1x+ b

# From our previous analysis, we got to know that out of 8 cont. numerical values, 4 of them have Strong rel, while others have moderate, So
# for accurate prediction - we will consider first four

#MSE - sum of (actual-predicted)^2/n
#R^2 - 1 - ((MSE of reg line)/(MSE of avg of the data line))


def SLR_highway():
    # Q1) in what way highway-L/100km can predict the price of the car
    lm = LinearRegression()
    x = x_train[['highway-L/100km']]
    y = y_train
    lm.fit(x, y)  # method of least squares - gives best fit line, from which we can extract the linear eq - ax + b
    intercept_of_line = lm.intercept_
    slope = lm.coef_
    print(f'\n Highway \n y = {slope}*x + {intercept_of_line}')  # 3455.52367659 x + (-14617.84305466459)

    yHat = lm.predict(x)
    for i in range(4, 11):
        print(f'for x = {i} the price is :  {yHat[i]}\n')

    Window1 = plt.figure(1)
    sns.regplot(x='highway-L/100km', y='price', data=df_c)
    plt.xlabel('highway-L/100km')
    plt.ylabel('price')
    plt.title('Regression plot between highway L per 100 km and price')
    #plt.savefig('Reg plot of Highway L per 100 km.png')
    # Model Evaluation Using Visualization
    # 1 - residual plot - plotting of residual points from regplot [ residual points - actual data - predicted data]
    # Why is that? Randomly spread out residuals means that the variance is constant,
    # and thus the linear model is a good fit for this data.
    Window2 = plt.figure(2)  # between highway-L/100km and price
    sns.residplot(x='highway-L/100km', y='price', data=df_c)
    plt.title('Residue plot')
    #plt.savefig('Residual plot - SLR-highway.png')
    plt.show()
    plt.close()
    # We can see from this residual plot that the residuals are not randomly spread around the x-axis,
    # leading us to believe that maybe a non-linear model is more appropriate for this data.
    r_score = lm.score(x, y)
    MSE = mse(y, yHat)
    return r_score, MSE


def SLR_engine():
    # so far not the best model, lets try another feature.- #engine size
    lm = LinearRegression()
    x = x_train[['engine-size']]
    y = y_train
    lm.fit(x, y)
    intercept_of_line_2 = lm.intercept_
    slope2 = lm.coef_
    print(f'\n Engine-Size \n  y = {slope2} *x + {intercept_of_line_2}')

    yHat = lm.predict(x)
    for i in range(100, 111, 1):
        print(f'for x = {i} the price is :  {yHat[i]}\n')

    Window1 = plt.figure(3)
    sns.regplot(x='engine-size', y='price', data=df_c)
    plt.xlabel('engine-size')
    plt.ylabel('price')
    plt.title('Regression plot between engine-size and price')
    #plt.savefig('Regression plot between engine-size and price.png')

    Window2 = plt.figure(4)
    sns.residplot(x='engine-size', y='price', data=df_c)
    plt.xlabel('engine-size')
    plt.ylabel('price')
    plt.title('Residue plot')
    #plt.savefig('Residual plot - SLR-engine.png')

    plt.show()
    plt.close()
    # better than the previous one
    r_score = lm.score(x, y)
    MSE = mse(y, yHat)
    return r_score, MSE


def SLR_horse():
    lm = LinearRegression()
    x = x_train[['horsepower']]
    y = y_train
    lm.fit(x, y)
    intercept_of_line3 = lm.intercept_
    slope3 = lm.coef_
    print(f'\n Horsepower \n y = {slope3}*x + {intercept_of_line3}')
    yHat = lm.predict(x)

    Window1 = plt.figure(5)
    sns.regplot(x='horsepower', y='price', data=df_c)
    plt.xlabel('Horsepower')
    plt.ylabel('price')
    plt.title('Regression plot between horsepower and price')
    #plt.savefig('Regression plot between horsepower and price.png')

    Window2 = plt.figure(6)
    sns.residplot(x='horsepower', y='price', data=df_c)
    plt.xlabel('Horsepower')
    plt.ylabel('price')
    plt.title('Residue plot')
    #plt.savefig('Residue plot - SLR-horsepower.png')

    plt.show()
    plt.close()
    r_score = lm.score(x, y)
    MSE = mse(y, yHat)
    return r_score, MSE


def SLR_city():
    lm = LinearRegression()
    x = x_train[['city-L/100km']]
    y = y_train
    lm.fit(x, y)
    intercept_of_line4 = lm.intercept_
    slope4 = lm.coef_
    print(f'\n City-L/100km \n y = {slope4}*x + {intercept_of_line4}')
    yHat = lm.predict(x)

    Window1 = plt.figure(7)
    sns.regplot(x='city-L/100km', y='price', data=df_c)
    plt.xlabel('City-L/100km')
    plt.ylabel('price')
    plt.title('Regression Plot between City L per 100 and price')
    #plt.savefig('Regression Plot between City L per 100 and price.png')

    Window2 = plt.figure(8)
    sns.residplot(x='city-L/100km', y='price', data=df_c)
    plt.xlabel('city-L/100km')
    plt.ylabel('price')
    plt.title('Residue plot')
    #plt.savefig('Residue plot - SLR-city.png')

    plt.show()
    plt.close()
    r_score = lm.score(x, y)
    MSE = mse(y, yHat)
    return r_score, MSE


# CONCLUSION : ALl Residue plots are not evenly spread - which indicates that Linear model will not give us accurate ans

# Moving towards Multiple Linear model as single Linear model is not suitable


def MLR():
    x2_5 = x_train[['horsepower', 'city-L/100km', 'engine-size', 'highway-L/100km']]
    y3 = y_train
    lm2 = LinearRegression()
    lm2.fit(x2_5, y3)
    intercept_of_line3 = lm2.intercept_
    slope3 = lm2.coef_
    print(f' y3 = {slope3[0]} *x2 + {slope3[1]} *x3 + {slope3[2]} *x4 + {slope3[3]} *x5  + {intercept_of_line3}')
    y3Hat = lm2.predict(x2_5)

    Window3 = plt.figure(9)
    ax0 = sns.regplot(x='horsepower', y='price', data=df_c, color='Yellow', label='horsepower')
    ax1 = sns.regplot(x='city-L/100km', y='price', data=df_c, ax=ax0, color='red', label='city-L/100km', marker='.')
    ax2 = sns.regplot(x='engine-size', y='price', data=df_c, ax=ax1, color='blue', label='engine-size', marker='*')
    ax3 = sns.regplot(x='highway-L/100km', y='price', data=df_c, ax=ax2, color='green', label='highway-L/100km',
                      marker='*')
    plt.xlabel('Horsepower, city-L/100km, engine-size and highway-L/100')
    plt.ylabel('price')
    plt.legend(loc='best')
    # not that clear, so that's why for MLR we use distribution plots (distribution of actual data vs predicted data)

    # Model Evaluation Using Visualization
    # 2 - distribution plots or density plots - lines over hist
    Window5 = plt.figure(10)
    ar0 = sns.distplot(x=df_c['price'], hist=False, color='red', label='Actual data')
    sns.distplot(x=y3Hat, color='blue', hist=False, label='Predicted data', ax=ar0)
    # hist is false, coz this plot is created from data points- data point falling in same category are taken and a histogram is created but
    # we want a distribution to understand the range of actual and predicted data

    plt.xlabel('price')
    plt.ylabel('proportions of cars')
    plt.title('Actual vs Predicted Data')
    plt.legend(loc='best')
    #plt.savefig('Actual vs Predicted (MLR) .png')
    plt.show()
    plt.close('all')
    # We can see that the fitted values are reasonably close to the actual values since the two distributions overlap a bit.
    # However, there is definitely some room for improvement.
    r_score = lm2.score(x2_5, y3)
    MSE = mse(y3, y3Hat)
    #cross-validation-score and predict - as this is the best model out of all
    R_val_score = cross_val_score(lm2, x_data[['horsepower', 'city-L/100km', 'engine-size', 'highway-L/100km']], y_data, cv=4)
    print(f'Cross-validation score [default - R^2 score] - {R_val_score} ')
    Cross_predictHat = cross_val_predict(lm2, x_data[['horsepower', 'city-L/100km', 'engine-size', 'highway-L/100km']], y_data, cv=4)
    print(f'Predictions using cross-validation - {Cross_predictHat[0:5]}')
    return r_score, MSE


# This may give us accurate prediction [ we will compare with other models using MSE and R^2 values ]

# as linear model was not suitable and their residual plot tells a non linear model will work
# so lets try their polynomial model

def Poly_plot_fig(model, independent_variable, dependent_variable, Name, start, end):
    in_put = np.linspace(start, end, 100)
    pred_out = model(in_put)
    plt.figure(11)
    plt.plot(independent_variable, dependent_variable, '.', in_put, pred_out, '-')
    plt.title('Polynomial Fit')
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    plt.savefig('')


def Poly_Reg_highway():
    # polynomial reg - variation of SLR and MLR - input variable has powers
    x = x_train['highway-L/100km']
    y = y_train
    fit_obj = np.polyfit(x, y, deg=8)
    eq = np.poly1d(fit_obj)
    Poly_plot_fig(eq, x, y, 'highway-L/100km', 4, 15)
    #plt.savefig('Polynomial plot- Highway')
    plt.show()
    plt.close()
    yHat = eq(x)
    r_score = r2_score(y, yHat)
    MSE = mse(y, yHat)
    return r_score, MSE


def Poly_Reg_engine():
    x = x_train['engine-size']
    y = y_train
    fit_obj = np.polyfit(x, y, deg=5)
    eq = np.poly1d(fit_obj)
    print(f'{fit_obj}\n {eq}')
    Poly_plot_fig(eq, x, y, 'engine-size', 50, 340)
    #plt.savefig('Polynomial plot- Engine')
    plt.show()
    plt.close()
    yHat = eq(x)
    r_score = r2_score(y, yHat)
    MSE = mse(y, yHat)
    return r_score, MSE


def Poly_Reg_horse():
    x = x_train['horsepower']
    y = y_train
    fit_obj = np.polyfit(x, y, deg=6)
    eq = np.poly1d(fit_obj)
    print(f'{fit_obj}\n {eq}')
    Poly_plot_fig(eq, x, y, 'horsepower', 45, 270)
    #plt.savefig('Polynomial plot- horsepower')
    plt.show()
    plt.close()
    yHat = eq(x)
    r_score = r2_score(y, yHat)
    MSE = mse(y, yHat)
    return r_score, MSE


def Poly_Reg_city():
    x = x_train['city-L/100km']
    y = y_train
    fit_obj = np.polyfit(x, y, deg=4)
    eq = np.poly1d(fit_obj)
    print(f'{fit_obj}\n {eq}')
    Poly_plot_fig(eq, x, y, 'city-L/100km', 4, 19.5)
    #plt.savefig('Polynomial plot- city')
    plt.show()
    plt.close()
    yHat = eq(x)
    r_score = r2_score(y, yHat)
    MSE = mse(y, yHat)
    return r_score, MSE


#Polynomial model of LSR is much better as it covers most of the data points

#Polynomial model of MLR is complex - IN THIS CASE, AS MLR is SUITABLE, so no need to check for polynomial version of it
#𝑌ℎ𝑎𝑡=𝑎+𝑏_1𝑋_1+𝑏_2𝑋_2+𝑏_3𝑋_1𝑋_2+𝑏_4𝑋_12+𝑏_5𝑋_22 (deg=2)

#Now that we have selected certain regression models, now it is time to Statistically compare and  choose the best one on the basis of
#R^2 value - lm.score() and MSE value - After this -  test the model with test data set [ as this step involves creating and training model]

#MSE - sum of (actual-predicted)^2/n
#R^2 - 1 - ((MSE of reg line)/(MSE of avg of the data line))




def main():
    R2_score_high, MSE_Value_high = SLR_highway()
    R2_score_engine, MSE_Value_engine = SLR_engine()
    R2_score_horse, MSE_Value_horse = SLR_horse()
    R2_score_city, MSE_Value_city = SLR_city()

    R2_score_multiple, MSE_Value_multiple = MLR()

    R2_score_high_poly, MSE_Value_high_poly = Poly_Reg_highway()  #works for deg - 8
    R2_score_engine_poly, MSE_Value_engine_poly = Poly_Reg_engine()   #works for deg - 5
    R2_score_horse_poly, MSE_Value_horse_poly = Poly_Reg_horse()    #works for deg - 6
    R2_score_city_poly, MSE_Value_city_poly = Poly_Reg_city()     #works for deg - 4

    print(f'\n Simple Linear Regression Model - Highway-L/100km \n R square score = {R2_score_high} \n Mean Square Error = {MSE_Value_high} ')
    print(f'\n Simple Linear Regression Model - Engine-size \n R square score = {R2_score_engine} \n Mean Square Error = {MSE_Value_engine} ')
    print(f'\n Simple Linear Regression Model - Horsepower \n R square score = {R2_score_horse} \n Mean Square Error = {MSE_Value_horse} ')
    print(f'\n Simple Linear Regression Model - City-L/100km \n R square score = {R2_score_city} \n Mean Square Error = {MSE_Value_city} ')

    print(f'\n Multiple Linear Model - Highway-L/100km, Engine-Size, Horsepower, City-L/100km \n '
          f'R square score = {R2_score_multiple} \n Mean Square Error = {MSE_Value_multiple}')

    print(f'\n Polynomial Regression Model - Highway-L/100km \n R square score = {R2_score_high_poly} \n Mean Square Error = {MSE_Value_high_poly} ')
    print(f'\n Polynomial Regression Model - Engine-size \n R square score = {R2_score_engine_poly} \n Mean Square Error = {MSE_Value_engine_poly} ')
    print(f'\n Polynomial Regression Model - Horsepower \n R square score = {R2_score_horse_poly} \n Mean Square Error = {MSE_Value_horse_poly} ')
    print(f'\n Polynomial Regression Model - City-L/100km \n R square score = {R2_score_city_poly} \n Mean Square Error = {MSE_Value_city_poly} ')


#------------------------------CONCLUSION-------------------------------------------------------------------------------

#Comparison between values - R^2 score should be highest and MSE value should be lowest
#Among linear models, Simple Linear Regression Model - Engine-size gives us the best result

#Now comparing it to Multiple Linear Model - Highway-L/100km, Engine-Size, Horsepower, City-L/100km, we see Multiple
#Linear Model gives us the best result

#Among Polynomial regression models, Polynomial Regression Model - Engine-size gives us the best result
#Comparing it with Simple Linear Regression Model - Engine-size, we see polynomial reg model gives better result

#Now Multiple Linear Model - Highway-L/100km, Engine-Size, Horsepower, City-L/100km VS Polynomial Regression Model -
#Engine-size, Multiple linear model gives us the best result

#So we can Say that The Multiple Linear Model - Highway-L/100km, Engine-Size, Horsepower, City-L/100km is the right
#regression model for us

#-----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
