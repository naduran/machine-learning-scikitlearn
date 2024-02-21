import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe()) #Reporte estadístico de los datos

    x = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']] #[[]] es operando sobre columnas en pandas
    y = dataset[['score']]
    print(x.shape)
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25) #25% de datos para validación

    model_linear = LinearRegression().fit(x_train, y_train)
    y_predict_linear = model_linear.predict(x_test)

    modelLasso = Lasso(alpha=1).fit(x_train, y_train) #Entre más grande el lambda más penalización
    y_predict_lasso = modelLasso.predict(x_test)

    modelRidge = Ridge(alpha=0.02).fit(x_train, y_train) #Entre más grande el lambda más penalización. 1 es defecto para Ridge
    y_predict_ridge = modelRidge.predict(x_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear loss: ", linear_loss)

    Lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso loss: ", Lasso_loss)

    Ridge_loss = mean_squared_error(y_test,y_predict_ridge)
    print("Ridge loss: ", Ridge_loss)

    print("="*32) # Cambios en coeficientes
    print("Coef Lasso") # Algunos serán 0
    print(modelLasso.coef_)

    print("Coef Ridge") # No serán 0 en total
    print(modelRidge.coef_)

    #ElasticNet
    print("*"*32) # Cambios en coeficientes

    modelElasticNet = ElasticNet(alpha=0.01).fit(x_train, y_train)
    y_predict_elasticNet = modelElasticNet.predict(x_test)

    ElasticNet_loss = mean_squared_error(y_test,y_predict_elasticNet)
    print("ElasticNet loss: ", ElasticNet_loss)

    print("Coef ElasticNet") # Algunos serán 0
    print(modelElasticNet.coef_)


'''
La regresión lineal es la que más se acerca a un buen modelo porque el error es muy cercano a cero, 
pero esto puede ser sobreentrenado conforme se agreguen más datos.

Ridge tuvo mejor renfdimiento que Lasso.

El feature de corruption fue eliminado de lasso, y tiene el peso más bajo en ridge

Un lambda más elevado en lasso quita el peso de más features a 0

Resultados para lambda lasso = 0.02 y alfa ridge = 1
Linear loss:  1.0069706535160003e-07
Lasso loss:  0.04407612502556149
Ridge loss:  0.004907623649802428
================================
Coef Lasso
[1.34852801 0.86631794 0.45206149 0.73937152 0.         0.22106631
 0.92404748]
Coef Ridge
[[1.09099957 0.95024554 0.85547248 0.87949524 0.66305766 0.75856148
  0.96891061]]

Resultados para lambda lasso = 1 y alfa ridge = 1
Linear loss:  7.913131459497546e-08
Lasso loss:  1.5879663731432112
Ridge loss:  0.009606023654125108
================================
Coef Lasso
[0. 0. 0. 0. 0. 0. 0.]
Coef Ridge
[[1.06897946 0.94457275 0.86426089 0.90861822 0.58406939 0.74988002
  0.94676787]]

Con respecto a ElasticNet se denota que el menjor resultado es con un alfa cercano a cero
(usando valores que vienen por defecto)

Esto sería similar a usar el modelo Ridge ya que alfa es cercano a cero. 

Resultados ElasticNet para alfa = 0.5

********************************
ElasticNet loss:  1.0554395819382383
Coef ElasticNet
[0.30691635 0.         0.         0.         0.         0.
 0.0682643 ]

Resultados ElasticNet para alfa = 0.1
********************************
ElasticNet loss:  0.18269066920373517
Coef ElasticNet
[1.13998382 0.52209825 0.34913463 0.03024606 0.         0.
 0.73785069]

Resultado ElasticNet para alfa = 0.9
********************************
ElasticNet loss:  1.0358696927223547
Coef ElasticNet
[0. 0. 0. 0. 0. 0. 0.]

Resultado ElasticNet para alfa = 0.01
********************************
ElasticNet loss:  0.010580839259343479
Coef ElasticNet
[1.10452294 0.90620757 0.85582604 0.88304935 0.46581088 0.69410932
 0.95162488]

 Resultado ElasticNet para alfa = 0.99
 ********************************
ElasticNet loss:  1.4847331992112702
Coef ElasticNet
[0. 0. 0. 0. 0. 0. 0.]

'''

