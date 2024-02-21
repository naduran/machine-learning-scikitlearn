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
- Instalar pip: python .\Recursos\get-pip.py
- Instalar entorno virtual (evitar conflicto de librerias): python -m pip install virtualenv
- Nombrar entorno virtual: python -m virtualenv entorno
- Activar entorno: entorno\Scripts\activate.bat (Si usas windows, asegurarse de que esté sobre la cmd y no powershell)
- Revisar archivo requirements.txt en la carpeta Recursos para instalación librerías (python -m pip install -r .\Recursos\requirements.txt)
- En caso de que falle reqirements.txt:
    - python -m pip install numpy
    - python -m pip install scipy
    - python -m pip install joblib
    - python -m pip install pandas
    - python -m pip install matplotlib
    - python -m pip install scikit-learn

- Verificar instalación correcta;
    - Iniciar interprete en la terminal: python
    - import sklearn
    - print(sklearn.__version__)
    - exit()