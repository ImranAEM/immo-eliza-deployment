from fastapi import FastAPI
from pydantic import BaseModel 
import joblib
import pandas as pd
import numpy as np


# Load trained Pipeline

pipeline = joblib.load("property_price_prediction_pipeline.joblib")



app = FastAPI(title = "Property Price Prediction API")

class HouseData(BaseModel):
    locality_name: str
    postal_code: int
    type_property: str
    subtype_property: str
    number_rooms: int
    living_area: int
    equipped_kitchen: str
    furnished: bool
    open_fire: bool
    terrace: bool
    garden: bool
    number_facades: int
    swimming_pool: bool
    state_building: str
    #terrace_surface: int





@app.get('/')

def home():
    return {'message' : 'Welcome to the Property Price Prediction API'}




@app.post('/predict/')
def predict_price(data: HouseData):

    # Convertir el objeto recibido en diccionario
    dictionary = data.model_dump()

    # Crear DataFrame a partir del diccionario
    df_house = pd.DataFrame([dictionary])


    # Convert Installed/Not installed to  0/1

    df_house['equipped_kitchen'] = df_house['equipped_kitchen'].apply(
        lambda x: 1 if x.lower() == "installed" else 0
    )

    

    # Renombrar columnas para que coincidan con el pipeline
    df_house.rename(columns={
        "locality_name": "Locality name",
        "postal_code": "Postal code",
        "type_property": "Type of property",
        "subtype_property": "Subtype of property",
        "number_rooms": "Number of rooms",
        "living_area": "Living area",
        "equipped_kitchen": "Equipped kitchen",
        "furnished": "Furnished",
        "open_fire": "Open fire",
        "terrace": "Terrace",
        "garden": "Garden",
        "number_facades": "Number of facades",
        "swimming_pool": "Swimming pool",
        "state_building": "State of building"
    }, inplace=True)




    # Añadir columnas que el pipeline espera y no vienen del usuario
    df_house["Terrace Surface"] = 0  # o el valor que corresponda

    
    prediction = pipeline.predict(df_house)


    return {"predicted price": f"€{prediction[0]:,.2f}"}
