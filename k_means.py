import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':
    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(10))

    #en aprenidzaje no supervisado no rquiere una variable target

    x = dataset.drop('competitorname', axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(x) # De a 8 datos se va ir formando el modelo
    print("Total de centros: ", len(kmeans.cluster_centers_)) # 4 centros configurados

    print('*'*64)
    print(kmeans.predict(x)) # Clasificación de cada dato

    dataset['group'] = kmeans.predict(x)
    print(dataset.head(10))