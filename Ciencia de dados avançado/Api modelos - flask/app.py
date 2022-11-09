from joblib import load

from flask import Flask, render_template, request, url_for, redirect, jsonify
import numpy as np

import pandas as pd



model = load('models/clf.joblib')



model1 = load('models/clf1.joblib')



app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST']) 

def predict():

	#request of all inputs

	features = [float(x) for x in request.form.values()]

	#data preparing

	features_array = [np.array(features)]

	print('features')

	print(features)

	print('features_array')

	print(features_array)

	#predict

	predictionslr = model.predict(features_array)

	predictionssvr = model1.predict(features_array)

	print(predictionslr)
	print(predictionssvr)

	prediction_lr = int(predictionslr[0])

	prediction_svr = int(predictionssvr [0])

	return render_template("home.html", predictionslr = "prediction on Linear Regression: {}".format(prediction_lr ), predictionssvr = "prediction on SVR: {}".format(prediction_svr))







if __name__ == '__main__':
	app.debug = True

	app.run()