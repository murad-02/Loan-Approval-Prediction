from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the best model (update the model filename if needed)
MODEL_DIR = os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_Logistic_Regression.joblib')
model = joblib.load(MODEL_PATH)

# User-friendly input fields (in order):
USER_FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area'
]

# Encoding maps (must match training)
GENDER_MAP = {'Male': 1, 'Female': 0}
MARRIED_MAP = {'Yes': 1, 'No': 0}
EDUCATION_MAP = {'Graduate': 0, 'Not Graduate': 1}
SELF_EMPLOYED_MAP = {'Yes': 1, 'No': 0}
PROPERTY_AREA_MAP = {'Rural': [1, 0, 0], 'Semiurban': [0, 1, 0], 'Urban': [0, 0, 1]}

# Helper to preprocess user input into model input
def preprocess_input(form):
    gender = GENDER_MAP.get(form.get('Gender', 'Male'), 1)
    married = MARRIED_MAP.get(form.get('Married', 'No'), 0)
    dependents = form.get('Dependents', '0')
    try:
        dependents = int(dependents)
    except:
        dependents = 0
    education = EDUCATION_MAP.get(form.get('Education', 'Graduate'), 0)
    self_employed = SELF_EMPLOYED_MAP.get(form.get('Self_Employed', 'No'), 0)
    applicant_income = float(form.get('ApplicantIncome', 0))
    coapplicant_income = float(form.get('CoapplicantIncome', 0))
    loan_amount = float(form.get('LoanAmount', 0))
    loan_amount_term = float(form.get('Loan_Amount_Term', 360))
    credit_history = float(form.get('Credit_History', 1))
    property_area = PROPERTY_AREA_MAP.get(form.get('Property_Area', 'Urban'), [0, 0, 1])
    # LoanAmount_Missing feature
    loanamount_missing = 1 if form.get('LoanAmount', '').strip() == '' else 0
    # Model expects: [Dependents, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, LoanAmount_Missing, Gender_encoded, Married_encoded, Education_encoded, Self_Employed_encoded, Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban]
    return [
        dependents, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, loanamount_missing, gender, married, education, self_employed
    ] + property_area

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = preprocess_input(request.form)
            arr = np.array(input_data).reshape(1, -1)
            pred = model.predict(arr)[0]
            prediction = 'Approved' if pred == 1 else 'Rejected'
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', features=USER_FEATURES, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
