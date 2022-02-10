import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------------------------------------------------------------------------
def loading_file():
    data_file_path = "C:\\Users\\Asus\\Downloads\\Cog =2\\imports-85.data"

    df_1 = pd.read_csv(data_file_path, header=None)  # as this file does not contain headers or Variables

    headers = (
        "Symboling [Risk Factor]", "normalized-losses", "make", "Fuel-type", "Aspiration", "num-of-doors", "Body-style",
        "Drive-wheels", "Engine-location", "Wheel-base", "length", "width", "height", "curb-weight", "engine-type",
        "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "Compression - ratio", "horsepower",
        "peak-rpm",
        "city-mpg", "highway-mpg", "price")

    df_1.columns = headers

    df_1.to_csv("Automobile.csv")
    df_1.to_csv("C:\\Users\\Asus\\Downloads\\Cog =2\\Automobile.csv")
    # print(f"Data \n \n {df_1}")
    # --------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------
    print(f"Type of Data of Dataframe - \n {df_1.dtypes}\n ")
    # print(f"Statistical Summary of the data of int type - \n{df_1.describe()}\n")
    # print(f"Statistical Summary of all types - \n{df_1.describe(include='all')}\n")
    # ---------------------------------------------------------------------------------------------------------
    return df_1


# ----------------------------------------Data Wrangling----------------------------------------------------------------

# 1) Identifying and Handling Missing Values

def Identifying_and_Handling_Values(df_1):
    df_1.replace("?", np.nan, inplace=True)
    missing = df_1.isnull()
    test = df_1.isnull().sum()  # another way to identify missing values
    print(test)
    # print(missing)
    for column in missing.columns:
        print(column)
        print(missing[column].value_counts())
        print(" ")

    # - Before replacing the data or dropping it, have to format data (because of data type mismatch) and after that replace it
    df_1['normalized-losses'] = df_1['normalized-losses'].astype("float")
    df_1['bore'] = df_1['bore'].astype("float")
    df_1['stroke'] = df_1['stroke'].astype("float")
    df_1['horsepower'] = df_1['horsepower'].astype("float")
    df_1['peak-rpm'] = df_1['peak-rpm'].astype("float")
    df_1['price'] = df_1['price'].astype("float")
    df_1['city-mpg'] = df_1['city-mpg'].astype("float")
    df_1['highway-mpg'] = df_1['highway-mpg'].astype("float")

    # - dropping NaN in price [ as it is out target, and predicted values can harm our analysis]
    df_1.dropna(subset=['price'], axis=0, inplace=True)
    # after dropping , we have to reset the index
    df_1.reset_index(drop=True, inplace=True)

    print(f"Type of Data of Dataframe - \n {df_1.dtypes}\n ")
    print(df_1)

    # Replacing Data

    avg_losses = df_1['normalized-losses'].mean()
    df_1['normalized-losses'].replace(np.nan, avg_losses, inplace=True)

    freq_door = df_1['num-of-doors'].mode()
    print(freq_door)
    df_1['num-of-doors'].replace(np.nan, 'four', inplace=True)

    bor_mean = df_1['bore'].mean()
    df_1['bore'].replace(np.nan, bor_mean, inplace=True)

    str_mean = df_1['stroke'].mean()
    df_1['stroke'].replace(np.nan, str_mean, inplace=True)

    hors_mean = df_1['horsepower'].mean()
    df_1['horsepower'].replace(np.nan, hors_mean, inplace=True)

    peak_mean = df_1['peak-rpm'].mean()
    df_1['peak-rpm'].replace(np.nan, peak_mean, inplace=True)

    print(f"\n Data After Handling Missing Values \n {df_1}")
    print(f"Descriptive analysis - \n {df_1.describe()}\n")

    return df_1


# 2) Data Formatting - Converting mpg units into Litre per 100 km or L/100km for Data Standardization
# - 1 mpg = 235.215 L/100km - L/100km = 235.215/mpg
def Data_Formatting_of_mpg(df_1):
    df_1['city-mpg'] = 235.215 / (df_1['city-mpg'])
    df_1.rename(columns={"city-mpg": 'city-L/100km'}, inplace=True)

    df_1['highway-mpg'] = 235.215 / (df_1['highway-mpg'])
    df_1.rename(columns={"highway-mpg": 'highway-L/100km'}, inplace=True)

    return df_1


# 3 - Data Normalization -  transforming data / Variables into similar range - so that they can similar effect on data and our analysis can be better
# Target = make range [0-1]
# we can use Special Feature Scaling Method {SFS} or Min-Max
# using SFS
def Data_Normalization_of_H_W_L(df_1):
    df_1['height'] = df_1['height'] / (df_1['height'].max())
    df_1['width'] = df_1['width'] / (df_1['width'].max())
    df_1['length'] = df_1['length'] / (df_1['length'].max())
    return df_1


# 4 - Data Binning - Making Categorical bins  from a Variable or data [ Making ranges]
# making bins of horsepower and price [ as low, mid and high ] type
# checking distribution with the help of histograms
def Data_Binning_of_Price_and_Horsepower(df_1):
    plt.figure(1)
    plt.hist(df_1['horsepower'])
    plt.xlabel('horsepower')
    plt.ylabel('Count')
    plt.title('Horsepower Bins')
    plt.savefig("Analysis for Binning of Horsepower")

    plt.figure(2)
    plt.hist(df_1['price'])
    plt.xlabel('price')
    plt.ylabel('Count')
    plt.title('price Bins')
    plt.savefig("Analysis for Binning of Price")

    plt.show(block=False)

    group_name = ['Low', 'Mid', 'High']
    bins_H = np.linspace(min(df_1['horsepower']), max(df_1['horsepower']), 4)
    print(bins_H)
    df_1['horsepower-binned'] = pd.cut(df_1['horsepower'], bins_H, labels=group_name, include_lowest=True)
    print(df_1[['horsepower', 'horsepower-binned']].head(20))
    print(df_1['horsepower-binned'].value_counts())

    bins_p = np.linspace(min(df_1['price']), max(df_1['price']), 4)
    print(bins_p)
    df_1['price-binned'] = pd.cut(df_1['price'], bins_p, labels=group_name, include_lowest=True)
    print(df_1[['price', 'price-binned']].head(20))
    print(df_1['price-binned'].value_counts())

    plt.figure(3)
    plt.hist(df_1['horsepower-binned'])
    plt.xlabel('Ranges')
    plt.ylabel('Count')
    plt.title('Horsepower-Binned_in-Ranges')
    plt.savefig('Horsepower Bins')

    plt.figure(4)
    plt.hist(df_1['price-binned'])
    plt.xlabel('Ranges')
    plt.ylabel('Count')
    plt.title('prices-Binned_in-Ranges')
    plt.savefig('Price Bins')

    plt.show(block=False)

    return df_1


# 5 - dummies - numerical representation of categorical data
# [ as certain model cannot understand words or string, only numbers]
def Dummies_Fuel_Type_and_asp(df_1):
    dummies_Fuel_Type = pd.get_dummies(df_1['Fuel-type'])
    # print(dummies_Fuel_Type.head())
    dummies_Fuel_Type.rename(columns={'gas': 'fuel-type_gas', 'diesel': 'fuel-type_diesel'}, inplace=True)
    print(dummies_Fuel_Type.head())

    df_1 = pd.concat([df_1, dummies_Fuel_Type], axis=1)
    print(df_1.head())
    df_1.drop('Fuel-type', axis=1, inplace=True)

    dummies_asp = pd.get_dummies(df_1['Aspiration'])
    # print(dummies_asp)
    dummies_asp.rename(columns={'std': 'Aspiration-std', 'turbo': 'Aspiration-turbo'}, inplace=True)
    print(dummies_asp.head())

    df_1 = pd.concat([df_1, dummies_asp], axis=1)
    df_1.drop('Aspiration', axis=1, inplace=True)

    return df_1


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    df = loading_file()
    df1 = Identifying_and_Handling_Values(df)
    df2 = Data_Formatting_of_mpg(df1)
    df3 = Data_Normalization_of_H_W_L(df2)
    df4 = Data_Binning_of_Price_and_Horsepower(df3)
    df_Car = Dummies_Fuel_Type_and_asp(df4)
    df_Car.to_csv("Cleaned_Automobile.csv")