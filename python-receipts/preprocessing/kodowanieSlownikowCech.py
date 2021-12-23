from sklearn.feature_extraction import DictVectorizer
import pandas as pd

data_dict = [{'czerwony': 2, 'niebieski': 4},
             {'czerwony': 4, 'niebieski': 3},
             {'czerwony': 1, 'zolty': 2},
             {'czerwony': 2, 'zolty': 2}]

dict_vectorizer = DictVectorizer(sparse=False)

dict_vectorizer.fit(data_dict)
features = dict_vectorizer.transform(data_dict)
print(features)
print(dict_vectorizer.get_feature_names_out())

dataframe = pd.DataFrame(features, columns=dict_vectorizer.get_feature_names_out())
print(dataframe)

doc1 = {'czerwony': 2, 'niebieski': 4}
doc2 = {'czerwony': 4, 'niebieski': 3}
doc3 = {'czerwony': 1, 'zolty': 2}
doc4 = {'czerwony': 2, 'zolty': 2}

doc_count = [doc1, doc2, doc3, doc4]

dict_vectorizer.fit(doc_count)g
x = dict_vectorizer.transform(doc_count)
print(dict_vectorizer.get_feature_names_out())

frame = pd.DataFrame(x, columns=dict_vectorizer.get_feature_names_out())
print(frame)

