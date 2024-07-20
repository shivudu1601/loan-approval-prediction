from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
with open('loanXG.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Assuming you have a scaler file saved as 'scalex.pkl'
with open('scalex.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=["POST"])
def submit():
    try:
        input_features = [float(x) for x in request.form.values()]
        input_feature = np.array(input_features).reshape(1, -1)

        names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        data = pd.DataFrame(input_feature, columns=names)

        # Debugging: Print input data
        print("Input Data:")
        print(data)

        # Scale the input data
        data_scaled = scaler.transform(data)

        # Debugging: Print scaled data
        print("Scaled Input Data:")
        print(data_scaled)

        prediction = model.predict(data_scaled)
        prediction = int(prediction[0])

        # Debugging: Print prediction
        print(f"Prediction: {prediction}")

        if prediction == 0:
            result_message = "Loan will Not be Approved"
        else:
            result_message = "Loan will be Approved"

    except ValueError as e:
        result_message = f"Invalid input: {e}. Please check your input and try again."
    except Exception as e:
        result_message = f"An error occurred: {e}. Please check your input and try again."

    return render_template("submit.html", result=result_message)


if __name__ == "__main__":
    app.run(debug=True)
