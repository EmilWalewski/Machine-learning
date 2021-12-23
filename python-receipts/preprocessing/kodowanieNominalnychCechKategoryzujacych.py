import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

feature = np.array([['Teksas'],
                   ['Kalifornia'],
                   ['Teksas'],
                   ['Delaware'],
                   ['Teksas']])


one_hot = LabelBinarizer()


print(one_hot.fit_transform(feature))

# output data
print(one_hot.classes_)

# reverse one hot transformation
print(one_hot.inverse_transform(one_hot.transform(feature)))


multiclass_feature = [('Teksas', 'Floryda'),
                      ('Kalifornia', 'Alabama'),
                      ('Teksas', 'Floryda'),
                      ('Delware', 'Floryda'),
                      ('Teksas', 'Alabama')]
one_hot_multiclass = MultiLabelBinarizer()
one_hot_multiclass.fit(multiclass_feature)
print(one_hot_multiclass.transform(multiclass_feature))
print(one_hot_multiclass.classes_)
print(one_hot_multiclass.inverse_transform(one_hot_multiclass.transform(multiclass_feature)))
