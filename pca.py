import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

#Regresión logística --> Clasificación
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def compute_accuracy(X_train, X_test, y_train, y_test, n_components, method):
    accuracies = []
    for n in n_components:
        if method == 'PCA':
            pca = PCA(n_components=n)
            pca.fit(X_train)
            X_train_transformed = pca.transform(X_train)
            X_test_transformed = pca.transform(X_test)
        elif method == 'IPCA':
            ipca = IncrementalPCA(n_components=n, batch_size=10)
            ipca.fit(X_train)
            X_train_transformed = ipca.transform(X_train)
            X_test_transformed = ipca.transform(X_test)
        else:
            raise ValueError("Invalid method. Use 'PCA' or 'IPCA'.")

        logistic = LogisticRegression()
        logistic.fit(X_train_transformed, y_train)
        accuracy = logistic.score(X_test_transformed, y_test)
        accuracies.append(accuracy)

    return accuracies

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

    print(x_train.shape)
    print(y_train.shape)

    #n_components = min(n_muestras, n_features)
    pca = PCA(n_components=0.9) # Se pueden usar decimales (Muestra mejor accuracy con 0.9 que con 3)
    pca.fit(x_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10) #ipca no manda todos los datos a entrenar al tiempo
    #No se pueden usar decimales
    ipca.fit(x_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    #Regresión logística para comparar los dos modelos
    logistic = LogisticRegression(solver='lbfgs') #Configuración por defecto

    dt_train = pca.transform(x_train)
    dt_test = pca.transform(x_test)

    logistic.fit(dt_train, y_train)
    print("SCORE PCA: ", logistic.score(dt_test, y_test)) #Vamos a seleccionar el accuracy

    dt_train = ipca.transform(x_train)
    dt_test = ipca.transform(x_test)

    logistic.fit(dt_train, y_train)
    print("SCORE IPCA: ", logistic.score(dt_test, y_test)) #Vamos a seleccionar el accuracy

################################################################################################
    # Gráfica variando el númeor de componentes en cada modelo
    n_components = range(2, 10)
    pca_accuracies = compute_accuracy(x_train, x_test, y_train, y_test, n_components, 'PCA')
    ipca_accuracies = compute_accuracy(x_train, x_test, y_train, y_test, n_components, 'IPCA')

    plt.plot(n_components, pca_accuracies, label='PCA')
    plt.plot(n_components, ipca_accuracies, label='IPCA')
    plt.title('N Components vs Accuracy - PCA vs IPCA')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy of Logistic Regression')
    plt.legend()
    plt.show()







