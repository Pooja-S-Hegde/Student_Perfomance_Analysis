from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model/student_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    feature_names = model_data['feature_names']

def get_grade(avg_score):
    """Convert average score to letter grade"""
    if avg_score >= 90:
        return 'A'
    elif avg_score >= 80:
        return 'B'
    elif avg_score >= 70:
        return 'C'
    elif avg_score >= 60:
        return 'D'
    else:
        return 'F'

def get_performance_message(avg_score, grade):
    """Get a descriptive message based on performance"""
    if grade == 'A':
        return "Excellent performance! Keep up the great work!"
    elif grade == 'B':
        return "Good performance! You're doing well."
    elif grade == 'C':
        return "Average performance. There's room for improvement."
    elif grade == 'D':
        return "Below average performance. Consider seeking additional help."
    else:
        return "Poor performance. Immediate intervention needed."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the scores from form
        math_score = float(request.form['math_score'])
        reading_score = float(request.form['reading_score'])
        writing_score = float(request.form['writing_score'])
        
        # Validate input ranges
        scores = [math_score, reading_score, writing_score]
        if any(score < 0 or score > 100 for score in scores):
            return render_template('result.html', 
                                 error="All scores must be between 0 and 100")
        
        # Make prediction
        input_data = np.array([[math_score, reading_score, writing_score]])
        predicted_avg = model.predict(input_data)[0]
        
        # Calculate actual average for comparison
        actual_avg = np.mean(scores)
        
        # Get grade and message
        predicted_grade = get_grade(predicted_avg)
        actual_grade = get_grade(actual_avg)
        message = get_performance_message(predicted_avg, predicted_grade)
        
        return render_template('result.html', 
                             math_score=math_score,
                             reading_score=reading_score,
                             writing_score=writing_score,
                             predicted_avg=round(predicted_avg, 1),
                             actual_avg=round(actual_avg, 1),
                             predicted_grade=predicted_grade,
                             actual_grade=actual_grade,
                             message=message)
    
    except ValueError:
        return render_template('result.html', 
                             error="Please enter valid numeric scores")
    except Exception as e:
        return render_template('result.html', 
                             error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
