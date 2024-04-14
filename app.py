from flask import Flask, render_template, request
import sqlite3
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained machine learning model for regression
regression_model = joblib.load('gradient_boosting_regressor_model.pkl')

# Load the trained machine learning model for classification
classification_model = joblib.load('final_classification_model.pkl')

# Load the DataFrame containing all feature names for regression
df_regression = pd.read_csv('preprocessed_dataregression.csv')

# Load the DataFrame containing all feature names for classification
df_classification = pd.read_csv('preprocessed_dataclassification.csv')

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/regression')
def regression():
    return render_template('regression.html')

@app.route('/classification')
def classification():
    return render_template('classification.html')

@app.route('/about_regression')
def about_regression():
    # Connect to the SQLite database within the route handler
    conn = sqlite3.connect('pd_titanic_house.db')
    # Fetch dataset information from the database
    df_info = pd.read_sql_query("SELECT * FROM pd_house LIMIT 10", conn)
    # Selecting only the first 10 columns
    df_info = df_info.iloc[:, :10]
    # Close the database connection after fetching data
    conn.close()
    return render_template('about_regression.html', dataset_info=df_info.to_html())

@app.route('/about_classification')
def about_classification():
    # Connect to the SQLite database within the route handler
    conn = sqlite3.connect('pd_titanic_house.db')
    # Fetch dataset information from the database
    df_info = pd.read_sql_query("SELECT * FROM pd_titanic LIMIT 10", conn)
    # Selecting only the first 10 columns
    df_info = df_info.iloc[:, :10]
    # Close the database connection after fetching data
    conn.close()
    return render_template('about_classification.html', dataset_info=df_info.to_html())


@app.route('/predict_regression', methods=['GET', 'POST'])
def predict_regression():
    if request.method == 'POST':
        # Get user input data
        user_input = [request.form.get(feat, 0) for feat in df_regression.columns[:10]]  # Get values for first 10 features
        user_input = [float(val) for val in user_input]  # Convert values to float
        user_input.extend([0.0] * (288 - 10))  # Fill remaining features with zeros
        # Make prediction using the trained model
        prediction = regression_model.predict([user_input])[0]
        # Format prediction result (if necessary)
        return render_template('prediction_regression_result.html', prediction=prediction)
    else:
        return render_template('predict_regression.html')

@app.route('/predict_classification', methods=['GET', 'POST'])
def predict_classification():
    if request.method == 'POST':
        # Get user input data
        user_input = [request.form.get(feat, 0) for feat in df_classification.columns[:10]]  # Get values for first 10 features
        user_input = [float(val) for val in user_input]  # Convert values to float
        user_input.extend([0.0] * (288 - 10))  # Fill remaining features with zeros
        # Make prediction using the trained model
        prediction = classification_model.predict([user_input])[0]
        # Format prediction result (if necessary)
        return render_template('prediction_classification_result.html', prediction=prediction)
    else:
        return render_template('predict_classification.html')
    
if __name__ == '__main__':
    app.run(debug=True)
