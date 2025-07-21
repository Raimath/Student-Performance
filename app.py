# Updated full code with model evaluation and overfitting check

from flask import Flask, request, render_template, jsonify, flash
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
from urllib.parse import quote
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Load dataset
df = pd.read_csv('dataset/csv.csv')

# Handle missing values
df[['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4']] = df[['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4']].fillna(0)
df['percentage'] = df[['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4']].mean(axis=1)
df = df.dropna(subset=['percentage'])

# Calculate dataset's mean percentage
average_percentage = df['percentage'].mean()

# Features & target
X = df.drop(columns=['name', 'percentage'])
y = df['percentage']

# Preprocessing
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

X_categorical = X.select_dtypes(include=['object'])
X_numerical = X.select_dtypes(include=['int64', 'float64'])

X_encoded = encoder.fit_transform(X_categorical)
X_scaled = scaler.fit_transform(X_numerical)

X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(X_categorical.columns))
X_scaled_df = pd.DataFrame(X_scaled, columns=X_numerical.columns)

X_processed = pd.concat([X_encoded_df, X_scaled_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.07, max_depth=7, min_samples_split=4, random_state=42)
model.fit(X_train, y_train)

# === MODEL EVALUATION SECTION START ===

# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print("Model Evaluation:")
# print("MAE:", round(mae, 2))
# print("MSE:", round(mse, 2))
# print("RMSE:", round(rmse, 2))
# print("R² Score:", round(r2, 2))

# train_score = model.score(X_train, y_train)
# test_score = model.score(X_test, y_test)
# print("Train R² Score:", round(train_score, 2))
# print("Test R² Score:", round(test_score, 2))

# cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='r2')
# print("Cross-Validation R² Scores:", cv_scores)
# print("Average CV R² Score:", round(cv_scores.mean(), 2))

# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, alpha=0.6, color='orange', edgecolors='black')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linestyle='--', linewidth=2)
# plt.xlabel('Actual Percentage')
# plt.ylabel('Predicted Percentage')
# plt.title('Actual vs Predicted Performance')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# === MODEL EVALUATION SECTION END ===

joblib.dump(model, 'model/trained_model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(average_percentage, 'model/average_percentage.pkl')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = {
            'gender': request.form['gender'],
            'parental level of education': request.form['education'],
            'test preparation course': request.form['test_prep'],
            'Participating in sports': request.form['sports'],
            'Subject 1': float(request.form['subject1']),
            'Subject 2': float(request.form['subject2']),
            'Subject 3': float(request.form['subject3']),
            'Subject 4': float(request.form['subject4'])
        }
    elif request.method == 'GET':
        input_data = {
            'gender': request.args.get('gender'),
            'parental level of education': request.args.get('education'),
            'test preparation course': request.args.get('test_prep'),
            'Participating in sports': request.args.get('sports'),
            'Subject 1': float(request.args.get('Subject1')),
            'Subject 2': float(request.args.get('Subject2')),
            'Subject 3': float(request.args.get('Subject3')),
            'Subject 4': float(request.args.get('Subject4'))
        }

    input_df = pd.DataFrame([input_data])

    encoder = joblib.load('model/encoder.pkl')
    scaler = joblib.load('model/scaler.pkl')
    average_percentage = joblib.load('model/average_percentage.pkl')

    X_categorical = input_df.select_dtypes(include=['object'])
    X_numerical = input_df.select_dtypes(include=['int64', 'float64', 'float32'])

    input_encoded = encoder.transform(X_categorical)
    input_scaled = scaler.transform(X_numerical)

    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(X_categorical.columns))
    input_scaled_df = pd.DataFrame(input_scaled, columns=X_numerical.columns)

    input_processed = pd.concat([input_encoded_df, input_scaled_df], axis=1)

    model = joblib.load('model/trained_model.pkl')
    predicted_percentage = model.predict(input_processed)[0]

    if input_data['gender'] == 'female':
        predicted_percentage += 5

    if input_data['test preparation course'] == 'completed':
        predicted_percentage += 5
    else:
        predicted_percentage -= 2

    education_boosts = {
        "master's degree": 4,
        "bachelor's degree": 3,
        "associate's degree": 2,
        "some college": 1,
        "high school": 0
    }
    predicted_percentage += education_boosts.get(input_data['parental level of education'], 0)

    if input_data['Participating in sports'] == 'no':
        predicted_percentage -= 2

    predicted_percentage = min(predicted_percentage, 100)

    feature_importances = model.feature_importances_
    contributions = feature_importances * input_processed.values[0]

    subject_columns = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4']
    other_columns = [col for col in input_processed.columns if col not in subject_columns]

    total_subject_score = sum([input_data[col] for col in subject_columns])
    
    if total_subject_score == 0:
        subjects_contributions = [0] * len(subject_columns)
    else:
        subjects_contributions = [
            (score / total_subject_score) * (predicted_percentage * 0.7)
            for score in [input_data[col] for col in subject_columns]
        ]

    other_total_contribution = predicted_percentage * 0.3
    other_contributions = contributions[[input_processed.columns.get_loc(col) for col in other_columns]]

    if np.sum(other_contributions) > 0:
        other_contributions = (other_contributions / np.sum(other_contributions)) * other_total_contribution

    labels = subject_columns + list(other_columns)
    importances = subjects_contributions + list(other_contributions)

    return jsonify({
        "prediction": round(predicted_percentage, 2),
        "average_score": round(average_percentage, 2),
        "labels": labels,
        "importances": importances
    })

    if request.method == 'POST':
        input_data = {
            'gender': request.form['gender'],
            'parental level of education': request.form['education'],
            'test preparation course': request.form['test_prep'],
            'Participating in sports': request.form['sports'],
            'Subject 1': float(request.form['subject1']),
            'Subject 2': float(request.form['subject2']),
            'Subject 3': float(request.form['subject3']),
            'Subject 4': float(request.form['subject4'])
        }
    elif request.method == 'GET':
        input_data = {
            'gender': request.args.get('gender'),
            'parental level of education': request.args.get('education'),
            'test preparation course': request.args.get('test_prep'),
            'Participating in sports': request.args.get('sports'),
            'Subject 1': float(request.args.get('Subject1')),
            'Subject 2': float(request.args.get('Subject2')),
            'Subject 3': float(request.args.get('Subject3')),
            'Subject 4': float(request.args.get('Subject4'))
        }

    input_df = pd.DataFrame([input_data])

    encoder = joblib.load('model/encoder.pkl')
    scaler = joblib.load('model/scaler.pkl')
    average_percentage = joblib.load('model/average_percentage.pkl')

    X_categorical = input_df.select_dtypes(include=['object'])
    X_numerical = input_df.select_dtypes(include=['int64', 'float64', 'float32'])

    input_encoded = encoder.transform(X_categorical)
    input_scaled = scaler.transform(X_numerical)

    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(X_categorical.columns))
    input_scaled_df = pd.DataFrame(input_scaled, columns=X_numerical.columns)

    input_processed = pd.concat([input_encoded_df, input_scaled_df], axis=1)

    model = joblib.load('model/trained_model.pkl')
    predicted_percentage = model.predict(input_processed)[0]

    if input_data['gender'] == 'female':
        predicted_percentage += 5

    if input_data['test preparation course'] == 'completed':
        predicted_percentage += 5
    else:
        predicted_percentage -= 2

    education_boosts = {
        "master's degree": 4,
        "bachelor's degree": 3,
        "associate's degree": 2,
        "some college": 1,
        "high school": 0
    }
    predicted_percentage += education_boosts.get(input_data['parental level of education'], 0)

    if input_data['Participating in sports'] == 'no':
        predicted_percentage -= 2

    predicted_percentage = min(predicted_percentage, 100)

    feature_importances = model.feature_importances_
    contributions = feature_importances * input_processed.values[0]

    subject_columns = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4']
    other_columns = [col for col in input_processed.columns if col not in subject_columns]

    total_subject_score = sum([input_data[col] for col in subject_columns])
    subjects_contributions = [(score / total_subject_score) * (predicted_percentage * 0.7)
                              for score in [input_data[col] for col in subject_columns]]

    other_total_contribution = predicted_percentage * 0.3
    other_contributions = contributions[[input_processed.columns.get_loc(col) for col in other_columns]]

    if np.sum(other_contributions) > 0:
        other_contributions = (other_contributions / np.sum(other_contributions)) * other_total_contribution

    labels = subject_columns + list(other_columns)
    importances = subjects_contributions + list(other_contributions)

    return jsonify({
        "prediction": round(predicted_percentage, 2),
        "average_score": round(average_percentage, 2),
        "labels": labels,
        "importances": importances
    })

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/add-data', methods=['GET', 'POST'])
def add_data():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        gender = request.form['gender']
        education = request.form['education']
        test_prep = request.form['test_prep']
        sports = request.form['sports']
        subject1 = int(request.form['subject1'])
        subject2 = int(request.form['subject2'])
        subject3 = int(request.form['subject3'])
        subject4 = int(request.form['subject4'])

        # For now, just print the data (you can save to CSV/DB here)
        print("Received Data:", name, gender, education, test_prep, sports, subject1, subject2, subject3, subject4)

        # Show success message
        return render_template('add_data.html', message='Data added successfully!')

    return render_template('add_data.html')



@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        flash("Thank you for contacting us! We'll get back to you soon.", "success")
    return render_template('contact.html')

@app.route('/dataset')
def dataset():
    def get_final_result(percentage):
        if percentage >= 75:
            return "Distinction"
        elif percentage >= 60:
            return "First Class"
        elif percentage >= 50:
            return "Second Class"
        else:
            return "Fail"

    # Add final result column
    df['final_result'] = df['percentage'].apply(get_final_result)

    # Move final_result right after percentage
    columns = list(df.columns)
    if 'final_result' in columns:
        columns.remove('final_result')
        percentage_index = columns.index('percentage')
        columns.insert(percentage_index + 1, 'final_result')
        df_reordered = df[columns]
    else:
        df_reordered = df

    # Generate HTML table with clickable names
    table_html = df_reordered.to_html(
        classes='styled-table',
        index=False,
        render_links=True,
        escape=False,
        formatters={
            'name': lambda name: f'<a href="/predict-page?name={quote(name)}&gender={quote(df.loc[df["name"] == name, "gender"].values[0])}&education={quote(df.loc[df["name"] == name, "parental level of education"].values[0])}&test_prep={quote(df.loc[df["name"] == name, "test preparation course"].values[0])}&sports={quote(df.loc[df["name"] == name, "Participating in sports"].values[0])}&subject1={df.loc[df["name"] == name, "Subject 1"].values[0]}&subject2={df.loc[df["name"] == name, "Subject 2"].values[0]}&subject3={df.loc[df["name"] == name, "Subject 3"].values[0]}&subject4={df.loc[df["name"] == name, "Subject 4"].values[0]}">{name}</a>'
        }
    )

    return render_template('dataset.html', table=table_html)

    table_html = df.to_html(
        classes='styled-table',
        index=False,
        render_links=True,
        escape=False,
        formatters={
            'name': lambda name: f'<a href="/predict-page?name={quote(name)}&gender={quote(df.loc[df["name"] == name, "gender"].values[0])}&education={quote(df.loc[df["name"] == name, "parental level of education"].values[0])}&test_prep={quote(df.loc[df["name"] == name, "test preparation course"].values[0])}&sports={quote(df.loc[df["name"] == name, "Participating in sports"].values[0])}&subject1={df.loc[df["name"] == name, "Subject 1"].values[0]}&subject2={df.loc[df["name"] == name, "Subject 2"].values[0]}&subject3={df.loc[df["name"] == name, "Subject 3"].values[0]}&subject4={df.loc[df["name"] == name, "Subject 4"].values[0]}">{name}</a>'
        }
    )
    return render_template('dataset.html', table=table_html)

if __name__ == '__main__':
    app.run(debug=True)

