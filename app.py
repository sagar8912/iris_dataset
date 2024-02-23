from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = load('model.pkl')
except Exception as e:
    print("Error loading the model:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
        
        # Make prediction
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Return prediction
        return render_template('index.html', prediction_text=f'Predicted class: {prediction[0]}')
    except Exception as e:
        print("Error making prediction:", e)
        return render_template('index.html', prediction_text='Error processing input')

if __name__ == '__main__':
    app.run(debug=True)

