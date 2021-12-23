import numpy as np
import pandas as pd

winequ = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')

# ustawienie indexu innego niz liczba np jakas unikalna wartosc w kolumnie

# winequ = winequ.set_index(winequ['fixed acidity'])

# print(winequ.loc['11.2']) wali errorem

print('---------------- DATAFRAME COUNT ----------------')
print(winequ.count())

print('---------------- DATAFRAME UNIQUE ----------------')
print(winequ['density'].unique())
print('---------------- DATAFRAME VALUE_COUNTS ----------------')
print(winequ['density'].value_counts())

print('---------------- DATAFRAME IS NULL ----------------')
print(winequ['density'].isnull())
print('---------------- DATAFRAME NO NULL ----------------')
print(winequ['density'].notnull())

print(winequ.groupby('density')['quality'].count())




