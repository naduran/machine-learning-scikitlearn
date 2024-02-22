import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe()) # min 0 y max 1 (datos binarios)

    x = dt_heart.drop(['target'], axis = 1) # Si quisieramos hacer el cambio sobre el mismo dt_heart, se debería usar inplace
    #como True en parámetros del drop
    y = dt_heart['target']
    x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

    knn_class = KNeighborsClassifier().fit(x_train, y_train)
    knn_pred = knn_class.predict(x_test)
    print('*'*64)
    print(accuracy_score(knn_pred, y_test))

    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators= 50).fit(x_train, y_train)
    bag_pred = bag_class.predict(x_test)
    print('*'*64)
    print(accuracy_score(bag_pred, y_test))

'''
Accuracy Score
****************************************************************
0.713091922005571 kmeans
****************************************************************
0.7409470752089137 bagging kmeans
'''