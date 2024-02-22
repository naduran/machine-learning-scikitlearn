import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score,
    KFold
)

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')

    x = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    model = DecisionTreeRegressor()
    #Prueba rápida
    score = cross_val_score(model, x, y, cv=3, scoring='neg_mean_squared_error') #cv es la cantidad de pliegues
    print(score)
    print(np.abs(np.mean(score)))

    #Más especializado
    kf = KFold(n_splits=3, shuffle=True, random_state=36)
    for train, test in kf.split(dataset):
        print(train)
        print(test)
        #Se puede implementar el modelo con cada uno de estos tests

