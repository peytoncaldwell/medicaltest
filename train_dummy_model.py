# train_dummy_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# 🔹 Step 1: Create dummy patient vitals data
# Example: heart rate, temperature, systolic BP, diastolic BP, outcome (0 = inpatient, 1 = discharge)

data = {
    'heart_rate': np.random.randint(60, 120, 100),
    'temperature': np.random.uniform(97, 103, 100),
    'systolic_bp': np.random.randint(90, 160, 100),
    'diastolic_bp': np.random.randint(60, 110, 100),
    'outcome': np.random.randint(0, 2, 100)
}

df = pd.DataFrame(data)

# 🔹 Step 2: Split features and labels
X = df[['heart_rate', 'temperature', 'systolic_bp', 'diastolic_bp']]
y = df['outcome']

# 🔹 Step 3: Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# 🔹 Step 4: Save the model to 'ml_model.pkl'
joblib.dump(model, 'ml_model.pkl')

print("✅ Model trained and saved as 'ml_model.pkl'")