import pandas as pd

from sklearn.linear_model import (
    RANSACRegressor,
    HuberRegressor,
)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad_corrupt.csv')
    print(dataset.head(5))

    x = dataset.drop(['country', 'score'], axis = 1) #Sacar las columnas que no aportan o que queremos predecir
    y = dataset[['score']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) #Si queremos replicabilidad usar random_state

    #Uso de diccionario
    #RANSAC es un metaestimador
    estimadores =  {
        'SVR' : SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER' : HuberRegressor(epsilon=1.35)
    }

    for name, estimador in estimadores.items():
        estimador.fit(x_train, y_train)
        predictions = estimador.predict(x_test)
        print("*"*64)
        print(name)
        print("MSE: ", mean_squared_error(y_test, predictions))

'''
****************************************************************
SVR
MSE:  1.0057029470281942
****************************************************************
RANSAC
MSE:  1.2780341730798587e-19
****************************************************************
HUBER
MSE:  6.99947262689187e-06

Al tomar un dataset con datos corruptos, RAMSAC y Huber muestran un mejor desempeño que SVM.
'''
