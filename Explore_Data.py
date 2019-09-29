import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd

#The most important part of the Pandas library is the DataFrame.
#A DataFrame holds the type of data you might think of as a table.
#this is similar to a sheet in Excel, or a table in a SQL database.
#As an example, we'll look at data about home prices in Melbourne, Australia.
#We'll apply the same processes to a new dataset, which has home prices in Iowa.

#Path of the file to read
iowa_file_path = './data/train.csv'

#Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

print(home_data.describe())

melbourne_file_path = './data/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

print(melbourne_data.columns)

'''
the Melb. data has some missing values.(some houses for which some variables weren't recorded.)
Iowa data doesn't have missing values in the columns we use.
so we'll take the simplest option and drop houses from our data
'''

#drop missing values(na = "not available")
melbourne_data = melbourne_data.dropna(axis=0)