'''
Building my own model
use scikit-learn library(for modeling the types of data typically stored in DataFrames)
 to create models

 1. Define: What type of model will it be? A decision tree? Some other type of model?
            Some other parameters of the model type are specified too.
 2. Fit: capture patterns from provided data. This is the heart of modeling.
 3. Predict
 4. Evaluate: Determine how accurate the model's predictions are.
'''
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = './data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

#Dot notation is used to select the "prediction target"
#use the dot notation to select the column we want to predict(prediction target)
#prediction target  = y
y = melbourne_data.Price

#Selecting with a column list, which we use to select the "features"
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

#define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

#Fit model
melbourne_model.fit(X, y)

'''
Many machine learning models allow some randomness in model training.
Specifying a number for random_state ensures you get the same results in each run.
This is considered a good practice.
You use any number, and model quality won't depend meaningfully 
on exactly what value you choose.

Now, we have a fitted model that we can use to make PREDICTIONS
'''
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are:")
print(melbourne_model.predict(X.head()))