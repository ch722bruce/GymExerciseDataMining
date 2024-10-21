import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("gym_members_exercise_tracking.csv")

# Display the first few rows
print(df.head(10))

# Dataset info
print(df.info())

# Label Encoding for categorical features
le = LabelEncoder()
df["Workout_Type"] = le.fit_transform(df["Workout_Type"])  # Encoding Workout Type
df["Gender"] = le.fit_transform(df["Gender"])  # Encoding Gender

# Display info after encoding
print(df.info())
print(df.head(3))

# Feature Selection and Target Separation
y = df["Calories_Burned"]  # Target
x = df.drop(columns=["Calories_Burned"])  # Features

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

# Model Training and Evaluation
models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
}

# Training each model and calculating the score
for name, model in models.items():
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"{name} score : {score:.2f}")

# Cross-Validation
for name, model in models.items():
    crossval = cross_val_score(model, x, y, cv=4)
    print(f"{name} Cross-Validation score: {np.mean(crossval):.2f}")

# Prediction using a sample
sample = df.sample()  # Selecting a random sample
print("Sample Data:")
print(sample)

# Removing the target column for prediction
sample_without_target = sample.drop(columns=["Calories_Burned"])

# Predicting the calories burned for the selected sample
predicted_calories = model.predict(sample_without_target)  # Predict using the trained model
print(f"Estimated calories burned for the sample: {predicted_calories[0]:.2f}")

# You can also manually input your own data for prediction:
# prediction = model.predict([[58, 0, 46.1, 1.67, 187, 129, 70, 1.28, 3, 25.3, 1.8, 4, 2, 16.53]])
# print(f"Estimated calories burned: {prediction[0]:.2f}")
