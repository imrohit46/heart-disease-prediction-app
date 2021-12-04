import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # input_data = [58,0,3,150,283,1,0,162,0,1,2,0,2]
    int_features = [float(x) for x in request.form.values()]
    final_input = [np.array(int_features)]
    prediction = model.predict(final_input)
    
    if prediction[0] == 0:
        response = 'Great, Your Heart is Healthy.'
    else:
        response = 'Uhh, You have a sign of Heart Disease.'

    return render_template('index.html', prediction_text=response)

if __name__ == "__main__":
    app.run(debug=True)