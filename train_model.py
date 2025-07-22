import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np

# Load data
df = pd.read_csv("StudentsPerformance.csv")

print("Dataset shape:", df.shape)
print("Dataset columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Create more intuitive features and target
# Let's predict final performance based on individual test scores
df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
df['average_score'] = df['total_score'] / 3

# Create grade categories based on average score
def get_grade(avg_score):
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

df['grade'] = df['average_score'].apply(get_grade)

# For prediction, we'll use current scores to predict overall performance
# This simulates predicting final grade based on current performance
X = df[['math score', 'reading score', 'writing score']]
y = df['average_score']

print(f"\nUsing features: {X.columns.tolist()}")
print(f"Predicting: average_score (range: {y.min():.1f} - {y.max():.1f})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Random Forest for better performance
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

# Save model and create a simple grade converter
model_data = {
    'model': model,
    'feature_names': X.columns.tolist()
}

with open("model/student_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\nModel saved successfully!")
print("The model now predicts overall academic performance based on:")
print("- Math Score (0-100)")
print("- Reading Score (0-100)") 
print("- Writing Score (0-100)")
