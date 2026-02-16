from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("motor_fault_model.pkl")

@app.route("/")
def home():
    return "Motor Fault Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]   # Get input JSON data
    
    features = np.array(data).reshape(1, -1)
    prediction = model.predict(features)
    
    return jsonify({
        "Prediction": prediction[0]
    })

if __name__ == "__main__":
    app.run(debug=True)
