from flask import Flask
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

# Importing the processed data
df = pd.read_csv('data/Processed_cardio.csv')

# Converting Pandas dataframe to Numpy array
X = df[['age', 'weight', 'gender', 'systolic_blood_pressure', 'diastolic_blood_pressure', 'cholesterol', 'glucose', 'physical_activity']].values
y = df['cardio_vascular_disease'].values

# AdaBoost Classifier algorithm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 11)
ada = AdaBoostClassifier(n_estimators= 100, random_state= 11).fit(X_train, y_train)

# Saving model to disk
pickle.dump(ada, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
