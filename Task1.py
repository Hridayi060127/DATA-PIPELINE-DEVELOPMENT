# ================================
# 1. IMPORT REQUIRED LIBRARIES
# ================================

import pandas as pd
import numpy as np

# Scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# For saving the pipeline
import joblib


# ================================
# 2. LOAD DATA
# ================================

# Replace with your actual dataset path
df = pd.read_csv("C:/Users/hrida/Downloads/archive/train.csv")

# Preview data
print("First 5 rows:")
print(df.head())


# ================================
# 3. SEPARATE FEATURES & TARGET
# ================================

# Change 'target' to your actual target column name
target_column = "pixel3"

X = df.drop(target_column, axis=1)  # Features
y = df[target_column]               # Target variable


# ================================
# 4. IDENTIFY COLUMN TYPES
# ================================

# Numerical columns (int, float)
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Categorical columns (object, category)
cat_cols = X.select_dtypes(include=["object", "category"]).columns

print("\nNumerical Columns:", list(num_cols))
print("Categorical Columns:", list(cat_cols))


# ================================
# 5. CREATE PREPROCESSING PIPELINES
# ================================

# ---- Numerical Pipeline ----
# Step 1: Fill missing values with mean
# Step 2: Scale features (important for ML models)

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


# ---- Categorical Pipeline ----
# Step 1: Fill missing values with most frequent value
# Step 2: Convert categories to numeric using OneHotEncoding

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ================================
# 6. COMBINE PIPELINES
# ================================

# Apply transformations to respective columns
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


# ================================
# 7. CREATE FULL PIPELINE (WITH MODEL)
# ================================

# Add a machine learning model at the end
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])


# ================================
# 8. TRAIN-TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ================================
# 9. TRAIN THE PIPELINE
# ================================

pipeline.fit(X_train, y_train)

print("\nModel training completed.")


# ================================
# 10. MAKE PREDICTIONS
# ================================

y_pred = pipeline.predict(X_test)


# ================================
# 11. EVALUATE MODEL
# ================================

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ================================
# 12. SAVE PIPELINE
# ================================

joblib.dump(pipeline, "data_pipeline.pkl")
print("\nPipeline saved as 'data_pipeline.pkl'")


# ================================
# 13. LOAD & REUSE PIPELINE
# ================================

# Example of loading and predicting on new data
loaded_pipeline = joblib.load("data_pipeline.pkl")

# Example: Predict on new/unseen data
# new_data = pd.read_csv("new_data.csv")
# predictions = loaded_pipeline.predict(new_data)

# print("\nPredictions on new data:", predictions)
