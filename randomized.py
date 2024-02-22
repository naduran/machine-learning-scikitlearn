import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':

    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset)

    x = dataset.drop(['country', 'rank', 'score'], axis = 1)
    y = dataset[['score']]

    reg = RandomForestRegressor()

    parametros = {
        'n_estimators': range(4,16), 
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': range(2,11)
    }
    # range(4,16) va hasta 15

    rand_est = RandomizedSearchCV(reg, parametros, n_iter =10, cv = 3, scoring= 'neg_mean_absolute_error').fit(x,y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    # RandomForestRegressor(criterion='absolute_error', max_depth=8, n_estimators=8)
    print(rand_est.predict(x.loc[[0]])) #Predicción del primer país

    



