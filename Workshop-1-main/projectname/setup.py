# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the Telco Customer Churn dataset
@st.cache_data
def load_data():
    file_path = "../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"  # Update path if needed
    df = pd.read_csv(file_path)

    # Convert 'TotalCharges' to numeric (handling empty values)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(df["TotalCharges"].median(), inplace=True)

    return df

df = load_data()

# Preprocess data
def preprocess_data(df):
    # Drop CustomerID (not useful for prediction)
    df.drop(columns=["customerID"], inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store for later use

    return df, label_encoders

df, label_encoders = preprocess_data(df)

# Define features and target
X = df.drop(columns=["Churn"])  # Features
y = df["Churn"]  # Target

# Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Streamlit UI
st.title("Customer Churn Prediction using Random Forest")

st.subheader("Dataset Overview")
st.write("The dataset contains customer information related to phone and internet services. Our goal is to predict whether a customer will churn (leave the service) or not.")

st.write(df.head())

# User input for hyperparameters
st.subheader("Train a Random Forest Classifier")
n_estimators = st.slider("Number of Trees (Estimators)", 10, 200, 100)
max_depth = st.slider("Maximum Depth of Trees", 1, 20, 10)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Display Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: **{accuracy:.2f}**")
st.write("Accuracy represents how well the model predicts whether a customer will churn or not.")

# Show classification report
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Visualizing the first Decision Tree in the Random Forest
st.subheader("Visualizing One Decision Tree in the Random Forest")
fig, ax = plt.subplots(figsize=(12, 8))
from sklearn.tree import plot_tree
plot_tree(model.estimators_[0], filled=True, feature_names=X.columns, class_names=["No", "Yes"], ax=ax)
ax.set_title("Decision Tree #1 in the Random Forest")
st.pyplot(fig)

# Display Predictions vs Actual Labels
st.subheader("Predictions vs Actual Labels")
predictions_df = pd.DataFrame({
    "Actual": y_test[:10].values,
    "Predicted": y_pred[:10]
})

st.write(predictions_df)
st.write("This table shows a sample of actual vs. predicted customer churn.")