# ==================================================================================================================================
# Imports

import numpy as np
import pickle
import subprocess
import os
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
    
# ==================================================================================================================================
# WebHook

# Path to repositorio and WSGI configuration
REPO_PATH = '/home/mailliwj/modelAPI'
SERVER_PATH = '/var/www/mailliwj_pythonanywhere_com_wsgi.py' 

@app.route('/webhook_2024', methods=['POST'])
def webhook():

    # Check request contains json data
    if not request.is_json:
        return jsonify({'Message': 'The request does not contain vlid JSON data'}), 400
    
    payload = request.json
        # Check payload has repo information
    if 'repository' not in payload:
        return jsonify({'Message': 'No repository information found in the payload'}), 400
    
    repo_name = payload['repository']['name']
    clone_url = payload['repository']['clone_url']
        
    # Try to change to repo directory
    try:
        os.chdir(REPO_PATH)
    except FileNotFoundError:
        return jsonify({'Message': F'The repo directory {REPO_PATH} does not exist'}), 404

    # Perform git pull
    try:
        subprocess.run(['git', 'pull'], check=True)
        subprocess.run(['touch', SERVER_PATH], check=True) # Reload PythonAnywhere WebServer
        return jsonify({'Message': f'Successfully pulled updates from the repository {repo_name}'}), 200
    
    except subprocess.CalledProcessError as e:
        return jsonify({'Message': f'Error during git pull: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)