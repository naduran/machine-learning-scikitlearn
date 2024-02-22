import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET']) #Probar con URL, en postman
    #Prueba con primera fila del datset para prediccion
def predict():
    X_text = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_text.reshape(1,-1))
    return jsonify({'prediccion': list(prediction)})

if __name__ == '__main__':
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080) #Los números más bajos de puerto pueden estar siendo usados por el sistema operativo para procesos internos

