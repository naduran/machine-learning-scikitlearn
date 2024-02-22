import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe()) # min 0 y max 1 (datos binarios)

    x = dt_heart.drop(['target'], axis = 1) # Si quisieramos hacer el cambio sobre el mismo dt_heart, se debería usar inplace
    #como True en parámetros del drop
    y = dt_heart['target']
    x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

    boost = GradientBoostingClassifier(n_estimators=50).fit(x_train, y_train)
    boost_pred = boost.predict(x_test)
    print('*'*64)
    print(accuracy_score(boost_pred, y_test))

'''
Accuracy Score
****************************************************************
0.9164345403899722 boosting
'''