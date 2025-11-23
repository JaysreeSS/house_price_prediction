# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model_rf.pkl', 'rb'))

def predict(regis, area, sqft, build, zone, bed, bath, commis, park, dist):
    inp = np.array([[regis, area, sqft, build, zone, bed, bath, commis, park, dist]]).astype(np.int32)
    prediction = model.predict(inp)
    output = int(prediction)

    st.success('Price of the House = Rs. {} (approx.)'.format(output))
    st.success('Amount per Square Feet = Rs. {} (approx.)'.format(int(output / sqft)))

# ------------------------------------------------------------------------
# UI  
# ------------------------------------------------------------------------

st.title("HOUSE PRICE PREDICTION")
st.subheader("An app that tells the price of your desired house!")

st.sidebar.title("DESCRIPTIONS")
st.sidebar.subheader("1. Area:")
st.sidebar.write("The area in which the house is located")
st.sidebar.subheader("2. Square Feet:")
st.sidebar.write("The square feet of the house")
st.sidebar.subheader("3. Number Of Bedrooms:")
st.sidebar.write("The number of bedrooms in the house")
st.sidebar.subheader("4. Number of Bathrooms:")
st.sidebar.write("The number of bathrooms in the house")
st.sidebar.subheader("5. Parking Facility:")
st.sidebar.write("Does the house have a parking facility?")
st.sidebar.subheader("6. Building Type:")
st.sidebar.write("What is the type of building of the house?")
st.sidebar.subheader("7. Zone:")
st.sidebar.write("The zone in which the house is located")
st.sidebar.write("[A = Agricultural, C = Commercial, I = Industrial, RH = Residential (High), RL = Residential (Low), RM = Residential (Medium)]")
st.sidebar.subheader("8. Distance from Main Road:")
st.sidebar.write("How far the house is from the main road?")
st.sidebar.subheader("9. Registration Fee:")
st.sidebar.write("What is the registration fee applicable after sales?")
st.sidebar.subheader("10. Commission:")
st.sidebar.write("What is the commission after sales?")

# ------------------------------------------------------------------------
# Inputs 
# ------------------------------------------------------------------------

area_list = ['Adyar', 'Anna Nagar', 'Chrompet', 'KK Nagar', 'Karapakkam', 'T Nagar']
area = st.selectbox("Area", area_list)
# Encode area according to model
area_map = {
    'Adyar': 0,
    'Anna Nagar': 1,
    'Chrompet': 2,
    'KK Nagar': 3,
    'Karapakkam': 4,
    'T Nagar': 5
}
area = area_map[area]

sqft = st.number_input("Square Feet of the house", min_value=100, max_value=10000)

bed = st.number_input("Number of Bedrooms (1 to 4)", min_value=1, max_value=4, step=1)

bath = st.number_input("Number of Bathrooms (1 or 2)", min_value=1, max_value=2, step=1)

park_status = st.radio("Parking Facility", ("Yes", "No"))
park = 1 if park_status == "Yes" else 0

build_status = st.radio("Building Type", ('Commercial', 'House', 'Other'))
if build_status == 'Commercial':
    build = 0
elif build_status == 'House':
    build = 1
else:
    build = 2

zone_status = st.radio("Zone", ('A', 'C', 'I', 'RH', 'RL', 'RM'))
zone_map = {'A': 0, 'C': 1, 'I': 2, 'RH': 3, 'RL': 4, 'RM': 5}
zone = zone_map[zone_status]

dist = st.text_input("Distance from Main Road (in meters)", max_chars=3, placeholder='Type here')
regis = st.text_input("Registration Fee (in rupees)", max_chars=7, placeholder='Type here')
commis = st.text_input("Commission (in rupees)", max_chars=7, placeholder='Type here')

# ------------------------------------------------------------------------
# Prediction Button 
# ------------------------------------------------------------------------
if st.button("Predict"):
    if dist == "" or regis == "" or commis == "":
        st.error("Please fill all the fields!")
    else:
        predict(int(regis), area, int(sqft), build, zone, bed, bath, int(commis), park, int(dist))
