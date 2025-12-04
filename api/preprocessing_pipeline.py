# Imports
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# Loading data
data = pd.read_csv('data/data.csv')


# Remove unnecessary columns 
data = data.drop(columns=['Property ID', 'Price per m2'])


# Create features and target variable
X = data.drop(['Price'], axis=1)

y = data['Price'] 


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)




# Split Categorical and Numerical Features
numerical_cols = []
categorical_cols = []


for X_column in X_train.columns:

    data_type = X_train[X_column].dtype

    if data_type in ['float64', 'int64']:
        numerical_cols.append(X_column)

    elif data_type == object:
        categorical_cols.append(X_column)


# Start the Pipeline w/ Encoding

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('scaler', StandardScaler())
])


categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


# Join the pipelines together (We connect the two preprocessing channels)

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])


# We put everything together, creating the final channeling (canalizacion final)
# We take our preprocessor and the output will go to our model

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])




# Train and predict Model

pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)



# Evaluate model accuracy

mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)

print(f"Model Performance: ")
print(f"R2 Score: {r2} ")
print(f"Root Mean Square Error: {rmse} ")
print(f"Mean Absolute Error: {mae} ")


# Save the pipeline to a new file

joblib.dump(pipeline, "property_price_prediction_pipeline.joblib")

# We have saved the channeling so that we can use it again in the future.