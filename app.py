from flask import Flask, render_template, request, url_for, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = model.predict(final)
    output = prediction
    return render_template('index.html', prediction_text= output)

if __name__ == '__main__':
    app.run(debug= True)