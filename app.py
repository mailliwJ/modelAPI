# ==================================================================================================================================
# Imports

import numpy as np
import pickle
from flask import Flask, jsonify, request

# ==================================================================================================================================

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return 'Welcome to my rain prediction app'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    required_params = ['pressure', 'sunshine', 'mean_temp']
    if not all(param in data for param in required_params):
        return jsonify({'Error': 'Missing required parameter. Must provide a value for all parameters'})
    
    try:
        pressure = request.args.get('pressure', None)
        sunshine = request.args.get('sunshine', None)
        mean_temp = request.args.get('mean_temp', None)

        input_data = np.array([[pressure, sunshine, mean_temp]])
        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)[0]

        return jsonify({'Prediction': prediction})
    
    except Exception as e:
        return jsonify({'Error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)