import streamlit as st
import joblib
import pandas as pd
import numpy as np




pipeline = joblib.load("api/property_price_prediction_pipeline.joblib")


st.title("Immo-Eliza - Price prediction")
st.write("Welcome to the property price prediction app")

st.image("https://cdn.pixabay.com/photo/2019/02/23/18/25/skyscraper-4016229_1280.jpg")





st.sidebar.subheader("Real Estate Price Estimator")
st.sidebar.image("https://cdn.pixabay.com/photo/2023/12/19/22/46/house-8458547_1280.jpg")

st.sidebar.text("This application allows you to estimate the price of a property "
"in Belgium using a machine learning model trained on real data. " \
"Enter the property features and receive an instant prediction.")




st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:
    
    type_property = st.selectbox('Type of Property', ['Appartment', 'House'])
    locality_name = st.text_input('Locality name', value='Ghent')
    postal_code = st.number_input('Postal code', min_value=1000, max_value=9999, value=9000)
    number_rooms = st.number_input('Number of rooms', min_value = 1, max_value = 100, value = 1)
    


with col2:

    living_area = st.number_input('Living area', min_value = 1, max_value = 1000, value = 50)
    equipped_kitchen = st.selectbox('Equipped kitchen', ['Installed', 'Not installed'])
    state_building = st.selectbox('State of building', ['Normal', 'Excellent', 'To be renovated'])
    number_facades = st.slider("Number of facades", min_value=0, max_value=4, value=2)




st.markdown("### Extra options")

terrace = st.checkbox("Terrace")
garden = st.checkbox("Garden")
furnished = st.checkbox("Furnished")
open_fire = st.checkbox("Open fire")
swimming_pool = st.checkbox("Swimming pool")





if type_property == "Appartment":
    subtype_property = "Appartment"  # adjust to the actual name of your dataset
else:
    subtype_property = "House"





# Convert Installed/Not installed to  0/1
if equipped_kitchen == "Installed":
    equipped_kitchen_num = 1
else:
    equipped_kitchen_num = 0



postal_code = int(postal_code)

furnished = int(furnished)
open_fire = int(open_fire)
terrace = int(terrace)
garden = int(garden)
swimming_pool = int(swimming_pool)



user_data = [{
 "Locality name": locality_name,
    "Postal code": postal_code,
    "Type of property": type_property,
    "Subtype of property": subtype_property,
    "Number of rooms": number_rooms,
    "Living area":living_area ,
    "Equipped kitchen": equipped_kitchen_num,
    "Furnished": furnished,
    "Open fire": open_fire,
    "Terrace": terrace,
    "Garden": garden,
    "Number of facades": number_facades ,
    "Swimming pool": swimming_pool,
    "State of building": state_building,
    "Terrace Surface": 0

}]




# Convert the dictionary into a pandas DataFrame
df_house = pd.DataFrame(user_data)




prediction = pipeline.predict(df_house)



if st.button("Predict Price"):
    try:
        prediction = pipeline.predict(df_house)
        st.success(f"The predicted price is: €{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")