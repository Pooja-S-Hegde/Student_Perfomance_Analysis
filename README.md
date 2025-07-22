 Student Performance Analysis & Prediction

A comprehensive data science project that predicts student academic performance using machine learning. This Flask-based web application analyzes student test scores and provides intelligent insights into overall academic performance.

 Project Overview

This project demonstrates end-to-end data science workflow including data analysis, feature engineering, machine learning model development, and web application deployment. The system predicts student performance grades based on individual subject scores using a Random Forest regression model.

 Dataset Information

- Source: Kaggle Student Performance Dataset
- Size: 1,000 student records
- Features: 8 columns including demographic and academic data
- Target Variable: Overall academic performance (derived from test scores)

 Dataset Columns:
- `gender`: Student gender
- `race/ethnicity`: Ethnic background (Groups A-E)
- `parental level of education`: Parents' highest education level
- `lunch`: Lunch program type (standard/free-reduced)
- `test preparation course`: Completion status of test prep
- `math score`: Mathematics test score (0-100)
- `reading score`: Reading comprehension score (0-100)
- `writing score`: Writing skills score (0-100)

 🔧 Technologies Used:

 Data Science & ML Stack:
- Python 3.12: Core programming language
- Pandas: Data manipulation and analysis
- NumPy: Numerical computations
- Scikit-learn**: Machine learning algorithms
- Random Forest Regressor: Primary ML model

 Web Development:
- Flask: Web framework for model deployment
- HTML/CSS: Frontend user interface
- Bootstrap 5: UI styling framework
- Jinja2: Template engine

 Key Features

 1. Intelligent Prediction System
- Predicts overall academic performance from individual subject scores
- Uses ensemble learning (Random Forest) for robust predictions
- Achieves 99.5% accuracy (R² = 0.995) on test data

 2. Interactive Web Interface
- Clean, responsive design with Bootstrap
- Real-time input validation
- Intuitive score input (0-100 range)
- Visual grade representation (A-F scale)

 3. Comprehensive Results Dashboard
- AI-predicted performance vs. actual average comparison
- Letter grade conversion with performance insights
- Subject-wise strength analysis
- Personalized feedback messages

4. Data Science Pipeline
- Automated feature engineering
- Model training and evaluation
- Cross-validation and performance metrics
- Model persistence using pickle



 Installation & Setup

 Prerequisites:
- Python 3.8 or higher
- pip package manager

 Steps:

1. Clone the Repository
   ```bash
   git clone https://github.com/Pooja-S-Hegde/Student_Perfomance_Analysis.git
   cd Student_Perfomance_Analysis
   ```

2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Train the Model
   ```bash
   python train_model.py
   ```

4. Run the Web Application
   ```bash
   python app.py
   ```

5. Access the Application
   - Open your browser and go to `http://127.0.0.1:5000`
   - Enter student test scores (Math, Reading, Writing)
   - Get instant performance predictions!

 🧠 Machine Learning Details

Model Architecture:
- Algorithm: Random Forest Regressor
- Estimators: 100 decision trees
- Features: Math, Reading, Writing scores
- Target: Predicted average performance score

 Performance Metrics:
- Mean Squared Error (MSE): 1.13
- R² Score: 0.995 (99.5% variance explained)
- Cross-validation: 80/20 train-test split

 Grade Classification:
- A Grade: 90-100 (Excellent)
- B Grade: 80-89 (Good)
- C Grade: 70-79 (Average)
- D Grade: 60-69 (Below Average)
- F Grade: <60 (Needs Improvement)

 Data Science Insights
   Key Findings:
1. Strong Correlation: High correlation between different subject scores
2.Predictive Power: Individual subject scores are excellent predictors of overall performance
3.Model Reliability: Random Forest handles non-linear relationships effectively
4. Feature Importance: All three subjects contribute significantly to predictions



