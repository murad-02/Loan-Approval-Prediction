# Loan-Approval-Prediction

A complete end-to-end machine learning project for predicting loan approval status using a real-world dataset. This project includes data analysis, preprocessing, feature engineering, model training, hyperparameter tuning, and a beautiful Flask web app for interactive predictions.

## Features
- Data cleaning, visualization, and feature engineering in Jupyter Notebook
- Multiple classification models with hyperparameter tuning and comparison
- Flask web app for user-friendly loan approval prediction
- Modern, responsive, and animated web UI
- Handles all preprocessing and encoding in Python (user inputs are human-friendly)

## Project Structure
```
├── assets/                # Project images and diagrams
├── Datasets/              # Training and test CSV files
├── Diagram/               # EDA and model comparison plots
├── Model/
│   ├── app.py             # Flask web application
│   └── templates/
│       └── index.html     # Web app HTML template
├── NotebookFile/
│   └── LoanApprovalPrediction.ipynb  # Main Jupyter notebook
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Flask app**
   ```bash
   cd Model
   python app.py
   ```
4. **Open your browser** and go to `http://127.0.0.1:5000/`

## Usage
- Fill in the loan application form with user-friendly values.
- Click **Predict** to see if the loan is likely to be approved.
- The app handles all preprocessing and encoding automatically.
- Use the **Reset** button to clear the form.

## Jupyter Notebook
- The notebook (`NotebookFile/LoanApprovalPrediction.ipynb`) contains all data analysis, preprocessing, model training, and evaluation steps.
- You can rerun the notebook to retrain or experiment with new models.

## Screenshots
See the `Diagram/` and `assets/` folders for EDA, model comparison, and UI screenshots.

## Credits
- Dataset: [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- UI: Custom, with Google Material Icons and modern CSS

---
For questions or contributions, please open an issue or pull request.