import streamlit as st
import pandas as pd
import joblib

# Load the KNN model
knn_model = joblib.load('./KNN_Model.pkl')

# App title
st.title("Heart Disease Prediction")

# Introduction
st.write("Welcome to the Heart Disease Prediction App! Please enter the following information to receive a prediction.")

# Define the input fields for each feature
st.header("Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=50, format="%d")
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=50, max_value=200, value=120)
chol = st.number_input("Serum Cholestoral (in mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0 = normal, 1 = ST-T wave abnormality, 2 = probable or definite left ventricular hypertrophy)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise relative to rest", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the peak exercise ST segment (0 = downsloping, 1 = flat, 2 = upsloping)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3) colored by fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)", [0, 1, 2])

# Create a button for prediction
if st.button("Predict"):
    # Prepare the data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Make the prediction
    prediction = knn_model.predict(input_data)[0]

    # Display the result
    st.header("Prediction Result")
    if prediction == 1:
        st.error("The model predicts that you **have** heart disease.")
    else:
        st.success("The model predicts that you **do not have** heart disease.")

# Additional Information
st.sidebar.header("More Information")
st.sidebar.write("""
    This app uses a K-Nearest Neighbors (KNN) model to predict the likelihood of heart disease based on various medical features. 
    Ensure that you enter accurate and relevant information for the best prediction results. 
    If you have any medical concerns, please consult with a healthcare professional.
""")
