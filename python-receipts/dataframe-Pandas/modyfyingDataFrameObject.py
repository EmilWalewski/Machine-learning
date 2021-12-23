import pandas as pd
import numpy as np

winequ = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')


print('---------------- REPLACE VALUES IN GIVEN COLUMN ----------------')
print(winequ['sulphates'].replace([0.56], ['NaN']).head())

print('---------------- RENAME GIVEN COLUMN ----------------')
print(winequ.rename(columns={'alcohol': 'Alco'}).head())

print('---------------- REPLACE VALUE TO NaN ----------------')
print(winequ['density'].replace([0.9978], [np.nan]).head())

print('---------------- REMOVE ROW FROM DATAFRAME ----------------')
print(winequ[winequ['sulphates'] != 0.56].head(20))

print('---------------- DROP DUPLICATES FROM SUBSET ----------------')
print(winequ.drop_duplicates(subset=['density'], keep='last').head(10))

print('---------------- CHECK IF ROW IS DUPLICATED; RETURN TRUE/FALSE ----------------')
print(winequ.duplicated())