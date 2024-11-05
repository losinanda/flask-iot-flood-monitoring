from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load model dari JSON dan H5
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

# API endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['input']).reshape((1, 24, 1))
    prediction = model.predict(input_data)
    prediction = prediction.tolist()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
