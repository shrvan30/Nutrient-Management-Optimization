from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

# Load the model, encoder, and scaler
model = joblib.load('fertilizer_price_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')  # Ensure scaler is saved and loaded

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/')
def home():
    return "Welcome to the Fertilizer Price Prediction API! Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json

        # Validate input structure
        if not data.get('categorical_features') or not data.get('numerical_features'):
            return jsonify({'error': 'Missing required data'}), 400

        # Extract features and reshape for processing
        categorical_features = np.array(data['categorical_features']).reshape(1, -1)
        numerical_features = np.array(data['numerical_features']).reshape(1, -1)

        # Normalize numerical features
        numerical_features = scaler.transform(numerical_features)

        # Encode categorical features
        encoded_features = encoder.transform(categorical_features)

        # Combine numerical and encoded categorical features
        full_features = np.hstack((numerical_features, encoded_features))

        # Predict using the model
        predicted_price = model.predict(full_features)

        # Dummy logic to predict fertilizer type (adjust this logic as per your model or dataset)
        fertilizer_types = ['Urea', 'DAP', 'MOP']
        predicted_fertilizer = fertilizer_types[int(predicted_price[0]) % len(fertilizer_types)]

        return jsonify({
            'predicted_price': float(predicted_price[0]),
            'predicted_fertilizer': predicted_fertilizer
        })

    except Exception as e:
        return jsonify({'error': f'Error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
