import pandas as pd
import joblib


# Load the trained pipeline
def load_pipeline(file_path):
    pipeline = joblib.load(file_path)

    return pipeline


# Convert input data into DataFrame
def convert_house_df(house_data):
    df_house = pd.DataFrame(house_data)

    return df_house


# Predict price directly using the pipeline
def predict_price(df_house, pipeline):
    prediction = pipeline.predict(df_house)
    
    return prediction






# ----------------------------
# Main
# ----------------------------

pipeline_path = "property_price_prediction_pipeline.joblib"

pipeline = load_pipeline(pipeline_path)

# "Equipped kitchen": "Installed"

house_data = [{
    "Locality name": "Aalst",
    "Postal code": 9300,
    "Type of property": "House",
    "Subtype of property": "Detached",
    "Number of rooms": 3,
    "Living area": 125,
    "Equipped kitchen": 1,
    "Furnished": 0,
    "Open fire": 0,
    "Terrace": 1,
    "Garden": 1,
    "Number of facades": 4,
    "Swimming pool": 0,
    "State of building": "Good",
    "Garden Surface": 90,
    "Terrace Surface": 15
}]


df_house = convert_house_df(house_data)

prediction = predict_price(df_house, pipeline)

print(f"Predicted price: {prediction[0]}€")
