from tkinter import W
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def main():
    # DATA READING
    data = pd.read_csv("./data/housing.csv")
    data.dropna(inplace=True)
    
    # train_data.hist(figsize=(15,8)) # shows distribution of data

    # NORMALIZATION
    data['total_rooms'] = np.log(data['total_rooms'] + 1)
    data['total_bedrooms'] = np.log(data['total_bedrooms'] + 1)
    data['population'] = np.log(data['population'] + 1)
    data['households'] = np.log(data['households'] + 1)
    
    data = data.join(pd.get_dummies(data.ocean_proximity)).drop(['ocean_proximity'], axis=1) # configures new columns for categorical feature
    
    
    # DATA ENGINEERING - adding features that may be useful
    data['bedroom_ratio'] = data['total_bedrooms'] / data['total_rooms']
    data['household_rooms'] = data['total_rooms'] / data['households']
    
    # DATA EXPLORATION

    # plt.figure(figsize=(12,6))
    # sns.scatterplot(x="latitude", y="longitude", data=data, hue="median_house_value", palette="coolwarm") # correlation between latitude/longitude and median value
    # plt.show()
    
    # plt.figure(figsize=(12,6))
    # sns.heatmap(data.corr(), annot=True, cmap="YlGnBu") # shows correlation between features
    # plt.show()
    
    # TRAINING
    scaler = StandardScaler()
    X = data.drop(['median_house_value'], axis=1)
    y = data['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.fit_transform(X_test)
    
    reg = LinearRegression()
    reg.fit(X_train_s, y_train)
    
    # EVALUATION
    print(reg.score(X_test_s, y_test))
    
    
    
if __name__ == "__main__":
    main()