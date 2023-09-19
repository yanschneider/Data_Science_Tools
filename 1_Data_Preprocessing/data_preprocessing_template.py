# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# missing data in each column
nan_count_per_column = dataset.isna().sum()

# select all columns with numbers:
imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# categorizing data and one hot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# specifiy what to do, the type of encoder, the column that is encoded and what to do with the other columns
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')

#numpy array is later required for training model
X = np.array(ct.fit_transform(X))

#Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#convert yes/no to 1/0
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#split into train and text matrices and arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# applying standardization (scale from 0 to 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# 3 is the number of categories
# fit will get mean and std dev - transform will calculate values betwenn 0 and 1 from this
X_train[:,3:] = sc.fit_transform(X_train[:,3:])





