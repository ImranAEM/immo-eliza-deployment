Deployment – Immo Eliza
---------------------------------------------------------------------------------------------
📌 Project Description
---------------------------------------------------------------------------------------------

This project is part of the BeCode BootCamp and focuses on deploying the Machine Learning model built in the previous project (Proyecto 05 – ML Model Immo Eliza).

The goal is to transform the trained model into a fully functional product available through:

A FastAPI REST API that receives property information and returns a price prediction.

A Streamlit Web Application where users can input property data through a friendly interface.

The project integrates model loading, preprocessing pipelines, backend API deployment, and frontend application development.

---------------------------------------------------------------------------------------------

🗂 Folder Structure
---------------------------------------------------------------------------------------------
    Immo-eliza-deployment/
    │
    ├─ api/
    │   ├─ predict.py
    │   ├─ preprocessing_pipeline.py
    │   ├─ property_price_prediction_pipeline.joblib
    │
    ├─ streamlit/
    │   └─ app.py
    │
    │
    ├─ requirements.txt
    └─ README.md
---------------------------------------------------------------------------------------------
📝 Main Features
---------------------------------------------------------------------------------------------
1. FastAPI – Backend / Prediction API

* Defines a /predict endpoint to receive property data as JSON.

* Loads the trained ML model and preprocessing pipeline.

* Validates inputs using Pydantic.

* Returns property price predictions formatted to two decimals.

2. Streamlit Web Application

* Interactive web interface for real-estate price prediction.

* Sends user inputs to the FastAPI backend.

* Displays the predicted price clearly and instantly.

* Fully deployed and publicly accessible.
---------------------------------------------------------------------------------------------

🔗 Streamlit Live App

The deployed web application can be accessed here:

👉 https://immo-eliza-genius.streamlit.app/
---------------------------------------------------------------------------------------------

3. Preprocessing Pipeline Integration

* Ensures the same transformations used during model training.

* Uses encoding and scaling to maintain consistency during predictions.

4. Deployment

* Streamlit app deployed online.

* API designed for local or cloud deployment.

* requirements.txt included for reproducibility.
---------------------------------------------------------------------------------------------

## 🧠 Learning Approach

This project emphasizes the real-world workflow of deploying machine learning models:

* Building and serving predictions through a REST API.

* Integrating backend and frontend tools.

* Structuring a clean and maintainable repository.

* Delivering a user-ready product.
---------------------------------------------------------------------------------------------

## ⏱️ Timeline

The project was completed over multiple sessions, covering:

* Pipeline and model loading

* API creation and testing

* Streamlit app development

* Deployment and integration

* Final adjustments and testing

---------------------------------------------------------------------------------------------

🏛️ Personal Situation

This project is part of the Immo Eliza Deployment assignment from the BeCode AI Bootcamp.
It extends the ML model created in the previous project and turns it into a functional application accessible to any user through an online platform.