import pandas as pd


dataframe = pd.DataFrame()

dataframe['name'] = ['bolek', 'name']
dataframe['surname'] = ['tw', 'surname']
dataframe['age'] = [12, '-']

print('---------------- CREATE DATAFRAME ----------------')
print(dataframe)

# append row
row = pd.Series(['Eric', 'Cartman', 20], index=['name', 'surname', 'age'])

dataframe = dataframe.append(row, ignore_index=True)

print('---------------- APPENDEN ROW TO DATAFRAME ----------------')
print(dataframe)


print('---------------- FILTER DATA IN DATAFRAME ----------------')
winequ = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
print(winequ[(winequ['fixed acidity'] > 8) & (winequ['quality'] == 6)].head())
