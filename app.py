# import necessary libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Print form data for debugging
        print(request.form)

        # Get data from the form
        rate = float(request.form.get('rate'))
        sales_in_first_month = float(request.form.get('sales_in_first_month'))
        sales_in_second_month = float(request.form.get('sales_in_second_month'))

        # Perform prediction using the loaded model
        prediction = model.predict([[rate, sales_in_first_month, sales_in_second_month]])

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except ValueError as e:
        return jsonify({'error': f'Invalid input. {str(e)}'}), 400


if __name__ == "__main__":
    app.run(debug=True)
