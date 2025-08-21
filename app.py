from flask import Flask, request, render_template, url_for, redirect, jsonify
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

with open('gender_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_model.pkl','rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST', 'GET'])
def fun():
    try:
        # Get input features from form
        features = [
            float(request.form['long_hair']),
            float(request.form['forehead_width_cm']),
            float(request.form['forehead_height_cm']),
            float(request.form['nose_wide']),
            float(request.form['nose_long']),
            float(request.form['lips_thin']),
            float(request.form['distance_nose_to_lip_long'])
        ]

        # Prepare input
        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Map prediction back to gender label
        gender = "Male" if prediction == 0 else "Female"

        return render_template("result.html", prediction_text=f"Predicted Gender: {gender}")
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)