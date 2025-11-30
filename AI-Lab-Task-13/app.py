from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Reverse mappings
gender_decode = {0: 'Female', 1: 'Male'}
grade_decode = {0: 'F', 1: 'A+', 2: 'A', 3: 'B', 4: 'C', 5: 'D', 6: 'E'}

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        try:
            age = int(request.form["age"])
            gender = int(request.form["gender"])
            study_hours = int(request.form["study_hours"])
            attendance = int(request.form["attendance"])
            test_score = int(request.form["test_score"])

            # Create input dataframe
            input_df = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "Study_Hours": [study_hours],
                "Attendance": [attendance],
                "Test_Score": [test_score]
            })

            # Scale input
            input_scaled = scaler.transform(input_df)

            # Predict grade
            pred = model.predict(input_scaled)[0]
            result = f"Predicted Grade: {grade_decode[int(pred)]}"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
