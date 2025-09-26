import streamlit as st
import pandas as pd
import joblib  # Using joblib instead of pickle
import os

# --- 1. LOAD THE TRAINED MODEL ---
MODEL_PATH = os.path.join("models", "final_model.pkl")

try:
    # Use joblib.load to open the model file
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at '{MODEL_PATH}'.")
    st.error("Please ensure the 'models' folder with your model file is in the same directory as 'app.py'.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.error("Please ensure the model was saved with 'joblib' and the library versions match.")
    st.stop()

# --- 2. DEFINE THE APP INTERFACE ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
st.title('🩺 Heart Disease Prediction App')
st.write("""
This app predicts the likelihood of a patient having heart disease based on their medical attributes.
Please enter the patient's details on the left sidebar and click 'Predict'.
""")

# --- 3. CREATE THE SIDEBAR FOR USER INPUT ---
st.sidebar.header('Patient Input Features')

def get_user_input():
    """Gets user input from the sidebar and returns it as a DataFrame."""

    # Input widgets
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholestoral in mg/dl (chol)', 100, 400, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ('True', 'False'))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', ('Normal', 'ST-T wave abnormality', 'Probable or definite left ventricular hypertrophy'))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 70, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', ('Yes', 'No'))
    oldpeak = st.sidebar.slider('ST depression induced by exercise (oldpeak)', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of the peak exercise ST segment (slope)', ('Upsloping', 'Flat', 'Downsloping'))
    ca = st.sidebar.selectbox('Number of major vessels colored by flourosopy (ca)', (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox('Thalassemia (thal)', ('Normal', 'Fixed defect', 'Reversible defect'))

    # --- Mappings: Convert user-friendly input to the format the model expects ---
    sex_map = {'Male': 1, 'Female': 0}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'True': 1, 'False': 0}
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Probable or definite left ventricular hypertrophy': 2}
    exang_map = {'Yes': 1, 'No': 0}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_map = {'Normal': 1, 'Fixed defect': 2, 'Reversible defect': 3}

    # Create a dictionary of the input data
    input_dict = {
        'age': age, 'sex': sex_map[sex], 'cp': cp_map[cp], 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs_map[fbs], 'restecg': restecg_map[restecg], 'thalach': thalach,
        'exang': exang_map[exang], 'oldpeak': oldpeak, 'slope': slope_map[slope], 'ca': ca, 'thal': thal_map[thal]
    }

    # Convert dictionary to a single-row pandas DataFrame
    features = pd.DataFrame(input_dict, index=[0])
    return features

# Get the input data from the user
input_df = get_user_input()

# Display the user's input for confirmation
st.subheader('Patient Input Summary')
st.write(input_df)

# --- 4. PREDICTION LOGIC AND DISPLAY ---
if st.button('**Predict**', type="primary"):
    try:
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Prediction Result')

        # Display result based on the prediction
        if prediction[0] == 1:
            st.error('**High Risk:** The model predicts this patient has a high likelihood of heart disease.')
            st.write(f"Confidence Score: **{prediction_proba[0][1]*100:.2f}%**")
        else:
            st.success('**Low Risk:** The model predicts this patient has a low likelihood of heart disease.')
            st.write(f"Confidence Score: **{prediction_proba[0][0]*100:.2f}%**")

        st.info("Disclaimer: This prediction is based on a machine learning model and should not be used as a substitute for a professional medical diagnosis.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")