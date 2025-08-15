import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model, feature columns, and the scaler
loaded_model = joblib.load('optimized_xgboost_model.pkl')
columns = joblib.load('model_columns.pkl')
scaler = joblib.load('BEST_Maize_yield_prediction_scaler.pkl')

# Get the list of soil types and numerical features
soil_columns = [col for col in columns if 'SOIL TYPE PERCENT1' in col]
soil_types = [col.replace('SOIL TYPE PERCENT1 (Percent)_', '') for col in soil_columns]
selected_soil_type = st.selectbox('Soil Type', soil_types)

state_columns = [col for col in columns if 'State Name' in col]
state_name = [col.replace('State Name_', '') for col in state_columns]
selected_state_name = st.selectbox('State Name', state_name)

numerical_features = [
    'MAIZE YIELD (Kg per ha)',
    'NITROGEN PER HA OF GCA (Kg per ha)',
    'PHOSPHATE PER HA OF GCA (Kg per ha)',
    'POTASH PER HA OF GCA (Kg per ha)',
    'AVERAGE RAINFALL (Millimeters)',
    'AVERAGE TEMPERATURE (Centigrate)',
    'AVERAGE PERCIPITATION (Millimeters)',
    'Year'
]
# Streamlit App Title and Description
st.title('Maize Yield Prediction Model')
st.markdown("### Predict the maize yield based on environmental and farming factors.")

# Create input widgets for user data
st.header('Input Parameters')
year = st.slider('Year', min_value=1966, max_value=2025, value=2023)
avg_temp = st.number_input('Average Temperature (Centigrate)', value=25.0)
nitrogen = st.number_input('Nitrogen per ha of GCA (Kg per ha)', value=10.0)
phosphate = st.number_input('Phosphate per ha of GCA (Kg per ha)', value=5.0)
potash = st.number_input('Potash per ha of GCA (Kg per ha)', value=3.0)
rainfall = st.number_input('Average Rainfall (Millimeters)', value=150.0)
precipitation = st.number_input('Average Precipitation (Millimeters)', value=100.0)
selected_soil_type = st.selectbox('Soil Type', soil_types)
selected_state_name = st.selectbox('State Name', state_name)

# Button to trigger the prediction
if st.button('Predict Maize Yield'):
    # Step 1: Feature Engineering on User Input
    # Step 2: Create a DataFrame from the user's input
    input_data = {col: 0 for col in columns}
    input_data['Year'] = year
    input_data['AVERAGE TEMPERATURE (Centigrate)'] = avg_temp
    input_data['NITROGEN PER HA OF GCA (Kg per ha)'] = nitrogen 
    input_data['PHOSPHATE PER HA OF GCA (Kg per ha)'] = phosphate  
    input_data['POTASH PER HA OF GCA (Kg per ha)'] = potash
    input_data['AVERAGE RAINFALL (Millimeters)'] = rainfall    
    input_data['AVERAGE PERCIPITATION (Millimeters)',] = precipitation
    
    soil_col_name = f'SOIL TYPE PERCENT1 (Percent)_{selected_soil_type}'
    if soil_col_name in input_data:
        input_data[soil_col_name] = 1
        
    state_col_name = f'State Name_{selected_state_name}'
    if state_col_name in input_data:
        input_data[state_col_name] = 1

    input_df = pd.DataFrame([input_data])
    input_df = input_df[columns]

    # Step 3: Standardize the numerical features in the new input DataFrame
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Step 4: Make the prediction
    try:
        prediction = model.predict(input_df)
        st.success(f'The predicted maize yield is: **{prediction[0]:.2f} Kg per ha**')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
