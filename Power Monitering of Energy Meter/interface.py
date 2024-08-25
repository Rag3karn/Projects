import streamlit as st
import pickle
import numpy as np

# Loading the trained model
with open('finalized_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Power Monitoring System')

# Input fields for 1x3 array
col1, col2, col3 = st.columns(3)
input1 = col1.text_input('Input 1:')
input2 = col2.text_input('Input 2:')
input3 = col3.text_input('Input 3:')

# Converting inputs to 2D array
if input1 and input2 and input3:
    user_input = np.array([[float(input1), float(input2), float(input3)]])
    prediction = model.predict(user_input)
    st.write(f'Prediction: {prediction}')
