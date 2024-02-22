import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == '__main__':
    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(5))

    x = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(x) #Parámetro ancho de banda por defecto
    print(meanshift.labels_)
    print(max(meanshift.labels_)) #Cantidad de grupos -1 

    print('*'*64)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_
    print('*'*64)	
    print(dataset)
    
    '''
    3 centros de arreglos

    ****************************************************************
[[2.25000000e-01 5.75000000e-01 1.00000000e-01 2.50000000e-02
  5.00000000e-02 2.50000000e-02 3.00000000e-01 1.00000000e-01
  5.50000000e-01 4.57599993e-01 3.67824996e-01 4.10442122e+01]
 [4.68750000e-01 5.00000000e-01 1.25000000e-01 1.56250000e-01
  9.37500000e-02 6.25000000e-02 1.25000000e-01 3.12500000e-01
  5.31250000e-01 4.57281243e-01 4.67874998e-01 5.21138597e+01]
 [8.26086957e-01 1.73913043e-01 3.04347826e-01 3.04347826e-01
  1.73913043e-01 1.73913043e-01 0.00000000e+00 5.21739130e-01
  4.34782609e-01 5.81391293e-01 6.38086963e-01 6.47120799e+01]]
    '''
