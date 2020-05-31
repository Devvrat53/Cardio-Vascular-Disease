from flask import Flask, render_template, request
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
    #print(int_features, len(int_features))
    final = [np.array(int_features)]
    #print(final)
    prediction = model.predict(final)
    output = round(prediction[0], 2)
    #print(output)
    if output == 1:
        return render_template('index.html', prediction_txt= 'You have a Cardio-vascular disease')
    else:
        return render_template('index.html', prediction_txt= 'You do not Cardio-vascular disease')
    
if __name__ == '__main__':
    app.run(debug= True)