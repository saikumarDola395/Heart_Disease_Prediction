from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model_path = "C:\\Users\\saiku\\OneDrive\\Desktop\\heart_disease_prediction\\heart_disease_prediction_model_percent.pkl"
with open(model_path, 'rb') as file:
    saved_model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability_of_heart_disease = None
    if request.method == "POST":
        # Get user input from the form
        user_input = list(map(int, request.form["features"].split(",")))
        
        # Predict
        user_input = np.array([user_input])
        prob = saved_model.predict_proba(user_input)
        probability_of_heart_disease = prob[0][1] * 100  # Probability of class 1
        
        # Convert prediction to human-readable format
        result = "Heart Disease" if prob[0][1] >= 0.5 else "No Heart Disease"
    
    return render_template("index.html", result=result, probability=probability_of_heart_disease)

if __name__ == "__main__":
    app.run(debug=True)
