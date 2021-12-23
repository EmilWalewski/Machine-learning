import numpy as np
import pandas as pd

dataframe = pd.DataFrame({'Score': ['niski', 'niski', 'sredni', 'sredni', 'wysoki', 'nieco wiecej niz sredni']})

scale_mapper = {'niski': 1,
                'sredni': 2,
                'nieco wiecej niz sredni': 2.1,
                'wysoki': 3}

print(dataframe['Score'].replace(scale_mapper))
