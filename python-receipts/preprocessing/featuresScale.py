import numpy as np
from sklearn import preprocessing

feature = np.array([[-500.0],
                    [-100.1],
                    [0],
                    [100.0],
                    [500.1]])
print(feature)

minmax_scale = preprocessing.MinMaxScaler(feature_range=[0, 1])

scaled_features = minmax_scale.fit_transform(feature)
print(scaled_features)

standard_scale = preprocessing.StandardScaler()

standarized = standard_scale.fit_transform(feature)
print(standarized)

print('------------- WRONG SCALE -------------')
# jezeli dane zawieraja elementy odstajace, moga miec one negatywny wplyw na standaryzacje
# poprzez wypaczenei sredniej i wariancji cechy.Zamiast przeskalowania lepiej uzyc
# mediany i rozstepu ćwiartowgo. klasa RobustScaler
x = np.array([[-1000.1],
              [-200.1],
              [500.5],
              [600.6],
              [9000.9]])
x2 = standard_scale.fit_transform(x)
print(x2)
print('------------- ROBUST SCALE -------------')
robust_scale = preprocessing.RobustScaler()
x3 = robust_scale.fit_transform(x)
print(x3)



