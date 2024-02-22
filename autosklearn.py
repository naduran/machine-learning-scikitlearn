''' 
Este es un ejecricio de prueba, esta librería funciona si tu máquina tiene los siguientes requerimientos:
* Se requiere un sistema operativo basado en Linux.
* Python (>=3.5) .
* Compilador para C++ (con soporte para C++11), por ejemplo GCC.
* SWIG (versión 3.0 o superior).
'''

#pip3 install auto-sklearn
import pandas as pd

import autosklearn.classification
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset)

    x = dataset.drop(['country', 'rank', 'score'], axis = 1)
    y = dataset[['score']]

    x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

    cls = autosklearn.classification.AutoSklearnClassifier().fit(x_train, y_train)

    predictions = cls.predict(x_test)

