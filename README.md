# machine-learning-scikitlearn

## Tipos de aprendizaje:
- Supervisado (Por observación)
- Aprendizaje por refuerzo (Menos información disponible) (Recompensa o castigo)
- No supervisado (Por descubrimiento) (Patrones)

## Otras ramas de la IA:
- Algoritmos evolutivos (Problema de optimización)
- Lógica difusa (Variables continuas)
- Agentes (Modelar un entorno que evalua interacción de diferentes agentes, contexto)
- Sistemas expertos (Sistemas de reglas para responder preguntas osbre los datos)

## Sickit-learn
- Limitado a la CPU, no herramienta de computer vision
- Clasificación [categorizar variables de salida], regresión [variable continua, no discretea (tiempo)], clustering [agrupación de datos y exploración de atípicos]

## Preparación del entorno
- Instalar pip: `python .\Recursos\get-pip.py`
- Instalar entorno virtual (evitar conflicto de librerias): `python -m pip install virtualenv`
- Nombrar entorno virtual: `python -m virtualenv entorno`
- Activar entorno: `entorno\Scripts\activate.bat` (Si usas windows, asegurarse de que esté sobre la cmd y no powershell)
- Revisar archivo requirements.txt en la carpeta Recursos para instalación librerías (`python -m pip install -r .\Recursos\requirements.txt`)
- En caso de que falle reqirements.txt:
    - `python -m pip install numpy`
    - `python -m pip install scipy`
    - `python -m pip install joblib`
    - `python -m pip install pandas`
    - `python -m pip install matplotlib`
    - `python -m pip install scikit-learn`

- Verificar instalación correcta:
    - Iniciar interprete en la terminal:`python`
    - `import sklearn`
    - `print(sklearn.__version__)`
    - `exit()`


## Impacto de features en modelos
- Los features son las variables de entrada
- Más features puede causar más ruido y generar más costo de procesamiento
- Sesgo y Varianza (Bias y Variance): Sesgo es que tan acertado es al mundo real. Varianza es que tan agrupados están.
- Se espera un sesgo y varianza bajos
- Overfitting: Sesgo bajo y varianza alta. Modeo muy complejo, no se ajuta con datos reales.
- Underfitting: Sesgo alto y varianza baja. Modelo muy simple.
-Técnicas:
    -Técnicas de reducción de dimensionalidad (PCA)
    - Regularización: Penzalizar los features que no estén aportando.
    - Balanceo: Oversampling y Undersampling

## PCA (Análisis de componentes principales)
- Usar cuando hay muchos features, alta correlacion entre features, overfitting o alto coste computacional.
- Combinar varios features que mantengan la información importante
- Pasos:
    - Matriz de covarianza (Que tanto se relacionan los features)
    - Hallar vectores propios y valores propios (fuerza y variabilidad de relaciones)
    - Los vectores propios con mayor variabilidad se escogen
- Variaciones:
    - IPCA. Se debe usar si se tiene un dataset exigente y pocos recursos
    - KPCA. Estructura no lineal separable (KERNEL)
- Uso: `python pca.py`
- Ejemplo de implementación PCA e IPCA sobre datset heart.csv con variación de número de componentes
 ![Descripción de la imagen](/Recursos/pca_ipca_batch11_random_27.jpg)

- KPCA
    - Kernels: Clasificador no lineal (tomar datos en una dimensión menor y se proyectan)
    - Lineales, polinomiales, gaussianos
    - `python kpca.py` (Recordar tener activo el entorno `entorno\Scripts\activate.bat`)

## Regularización
- Aplicar penalizaciones a las variables que no aportan
- Disminuir complejidad del modelo (Overfitting)
- Más sesgo por menos varianza
- Pérdida: Que tan lejos estamos de los datos reales
- Menos pérdida, mejor modelo
- Tipos:
    - L1 Lasso. Volver 0 los features que más ruido producen (mínimos cuadrados) (En la lambda se hace la penalización)
    - L2 Ridge. Les quita valor a los features que menos aportan, pero permite que sigan en el modelo ya que no llegan a 0. (Se penaliza con la pendiente al cuadrado y no valor absoluto como en L1)
    - ElasticNet. Combinación de los anteriores
    - Usar Lasso cuando hay pocos features que se relacionen con la variable a predecir
    - Usar Ridge si hay varios features relacionados con la variable a predecir
    - Parámetro alfa en ElasticNet: Si es cercano a cero se comporta como Ridge, su es cercano a 1 se comporta como Lasso
    - `python regularizacion.py` (Recordar tener activo el entorno `entorno\Scripts\activate.bat`)

## El problema de valores atípicos
- Cualquier medición que no se comporta como el patrón de los otros
- Pueden generar sesgos, detección temprana de fallos
- Métodos:
    - Estadísticos:
        - Z-Score: Desviación estándar, distancia de un punto a la media (cuantas desviaciones)
        - DBSCAN: Agrupaciones de los datos
        - Fórmula:
            - Si el punto está por debajo del cuartil1 (Q1) menos 1.5 del valor del rango intercuartílico (Q3-Q1)
            - Si el punto está sobre el cuartil3 (Q2) más 1.5 del valor del rango intercuartílico (Q3-Q1)
            - Boxplots, gráficas de distribución de una variable entre sí, teniendo en cuenta sus extremos y centro

## Regresiones Robustas con Scikit-learn
- RANSAC: Muestreo aleatorio sobre los datos. Se asume que los datos selecciónados son los buenos. Se entrena el modelo y se compara con otros datos seleccionados (varios muestreos). Encuentra la mejor combinación (inliners)
- Huber Regressor: Disminuye el aporte de los valores atípicos en el modelo. A partir de esto calcula la pérdida-
    - Se revisa si el error absoluto de la pérdida está sobre un umbral (epsilon)
    - El valor más recomendado de epsilon es 1.35 (95% eficiencia)
    - `python robust.py` (Recordar tener activo el entorno `entorno\Scripts\activate.bat`)

## Métodos de ensamble
- Probar diferentes estimadores y aplicar un método para lograr un consenso
- Diversidad
- Estrategias:
    - Bagging. La opinión de varios expertos en paralelo y lograr un consenso
    - Bootstrap Aggragation
        - Se crean particiones uniformes
        - Se construyen conjuntos independientes
        - Se elige una respuesta según los resultados individuales
        - Random forest, Voting Classifiers/Regressors
    - Boosting. Un experto aprende del error anterior de otro experto y lo va mejorando
        - Clasificadores más pequeños, fortalecer a través de la repetición
        - Termina en consenso
        - AdaBoost, Gradient Tree Boosting, XGBoost (No está en sklearn)
    - `python bagging.py` (Recordar tener activo el entorno `entorno\Scripts\activate.bat`)
    - `python boosting.py` (Recordar tener activo el entorno `entorno\Scripts\activate.bat`)
    - boosting demuestra mejores resultados que bagging

## Estrategias de Clustering
- Identificar patrones que no son visibles
- No supervisado, no etiquetas
- Casos de uso:
    - Sabemos cuantos grupos queremos (k) (k-means, spectral Clustering)
    - Preferimos que el modelo nos de la (k) óptima (Meanshift,Clustering jerarquico,, DBScan)
    - Se puede agregar estas columnas con el grupo a los datasets



    
    






