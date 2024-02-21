import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart.head(5))

    dt_features = dt_heart.drop(['target'], axis = 1) # Sin la columna que queremos clasificar
    dt_target = dt_heart['target'] 

    #Normalización de datos
    dt_features = StandardScaler().fit_transform(dt_features)

    #Partir el conjunto de entrenamiento
    x_train, x_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=27) 
    #Añadir random_state para replicabilidad

    kpca = KernelPCA(n_components=4, kernel='poly')
    # lineal: PCA normal
    # poly: Polinomial
    #rbf: Gaussiano
    kpca.fit(x_train)

    dt_train = kpca.transform(x_train)
    dt_test = kpca.transform(x_test)

    #Luego de reducir dimensionalidad de los datos
    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(dt_train, y_train)
    print("SCORE KPCA: ", logistic.score(dt_test, y_test))



