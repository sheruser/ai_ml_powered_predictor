from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from model import load_model, predict
from api import configure_gemini, ask_gemini

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load ML model
ml_model = load_model()
configure_gemini(os.getenv("GEMINI_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    # Get form data
    time_spent_alone = float(request.form['time_spent_alone'])
    stage_fear = request.form['stage_fear']
    social_event = float(request.form['social_event'])
    going_outside = float(request.form['going_outside'])
    drained = request.form['drained']
    friend_circle = float(request.form['friend_circle'])
    post_frequency = float(request.form['post_frequency'])
    
    # Prepare data dictionary
    data = {
        "Time_spent_Alone": time_spent_alone,
        "Stage_fear": 1 if stage_fear == "Yes" else 0,
        "Social_event_attendance": social_event,
        "Going_outside": going_outside,
        "Drained_after_socializing": 1 if drained == "Yes" else 0,
        "Friends_circle_size": friend_circle,
        "Post_frequency": post_frequency
    }
    
    # Get prediction
    prediction = predict(ml_model, data)
    personality = 'Extrovert' if prediction == 1 else 'Introvert'
    
    # Get Gemini explanation
    gemini_response = ask_gemini(prediction, data)
    
    # Return results page
    return render_template(
        'result.html',
        personality=personality,
        explanation=gemini_response,
        prediction=prediction,
        data=data
    )

if __name__ == '__main__':
    app.run(debug=True)