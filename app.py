from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app=Flask(__name__)

@app.route("/", method=['GET'])
def homepage():
    return render_template("index.html")

@app.route("/train", method=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful"

@app.route("/predict", method=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])

            data = [fixed_acidity,volatile_acidity]
            data = np.array(data).reshape(1,11)

            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))
        
        except Exception as e:
            raise e
        
    else:
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)