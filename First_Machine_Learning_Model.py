#step1: SPECIFY Prediction Target
#select target variable, which corresponds to the sales price.
#save this as variable 'y'
#print a list of the columns to find the name of the column that we need
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#Path of the file to read
iowa_file_path = './data/train.csv'

home_data = pd.read_csv(iowa_file_path)

#print(home_data.columns)

#Specifying Prediction Target
y = home_data.SalePrice
#print(y)

#Creating a DataFrame 'X' that will hold the predictive features
#You'll use just the following columns in the list (you can copy and paste the whole list to save some typing, though you'll still need to add quotes):
#    * LotArea
#    * YearBuilt
#    * 1stFlrSF
#    * 2ndFlrSF
#    * FullBath
#    * BedroomAbvGr
#    * TotRmsAbvGrd

#first, make a list
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#then, we assign the list to X via home_data
X = home_data[feature_names]
print(X.head())

#Specify and Fit Model
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)

#make predictions
predictions = iowa_model.predict(X)
print(predictions)