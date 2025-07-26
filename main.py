from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import skops.io
import os

MODEL_FILE = "model.skops"
PIPELINE_FILE = "pipeline.skops"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown='ignore')),
    ])
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing.csv")
    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0, 1.5, 3.0, 4.5, 6.0, float("inf")],
                                   labels=[1, 2, 3, 4, 5])
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop('income_cat', axis=1).to_csv("input.csv", index=False) 
        housing = housing.loc[train_index].drop('income_cat', axis=1)  
    
    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis=1)
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]
    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)
    skops.io.dump(model, MODEL_FILE)
    skops.io.dump(pipeline, PIPELINE_FILE)
    print("Model and pipeline saved.")
else:
    model_file = skops.io.get_untrusted_types(file=MODEL_FILE)
    pipeline_file = skops.io.get_untrusted_types(file=PIPELINE_FILE)
    model = skops.io.load(MODEL_FILE, trusted=model_file)
    pipeline = skops.io.load(PIPELINE_FILE, trusted=pipeline_file)
    
    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.fit_transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions
    input_data.to_csv("output.csv", index=False)
    print("Model loaded and predictions saved to output.csv.")