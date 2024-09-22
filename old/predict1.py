import pickle
from flask import Flask
from flask import request
from flask import jsonify


with open('model1.bin', 'rb') as file:
    model = pickle.load(file)
with open('dv.bin', 'rb') as file:
    dv = pickle.load(file)

app = Flask('scoring')

@app.route('/predict', methods=['POST'])
def predict_score():
    customer = request.get_json()
    vectorized = dv.transform(customer)
    y_pred = model.predict_proba(vectorized)[0,1]
    result = {
        'predicted_score': float(y_pred)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
