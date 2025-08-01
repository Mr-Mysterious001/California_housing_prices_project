import pandas as pd
import numpy as np
from main import build_pipeline
import matplotlib.pyplot as plt 
from sklearn.metrics import root_mean_squared_error

data = pd.read_csv("housing.csv")
input_data = pd.read_csv("input.csv")
output_data = pd.read_csv("output.csv")
num_attribs = data.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]
pipe = build_pipeline(num_attribs, cat_attribs)
pipe.fit_transform(data)

#Visualizing the data using graphs in matplotlib

features = ['median_income', 'housing_median_age', 'population', 'median_house_value']
pd.plotting.scatter_matrix(data[features],c=data['median_house_value'], cmap='viridis', alpha=0.5, figsize=(8, 8), diagonal='hist')
plt.suptitle('Scatter Matrix of Selected Features')
plt.grid(True)
plt.savefig("images/scatter_matrix.png")
plt.close()


plt.figure()
plt.scatter(data['longitude'], data['latitude'], c=data['median_house_value'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Median House Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Distribution of Median House Value')
plt.grid(True)
plt.savefig("images/Lat_Long.png")  # give meaningful names if multiple plots
plt.close()


rmse_value = root_mean_squared_error(input_data['median_house_value'], output_data['median_house_value'])
plt.figure()
plt.plot(input_data['median_house_value'], output_data['median_house_value'], 'o', markersize=5, alpha=0.5, markerfacecolor='green')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title(f'Actual vs Predicted Median House Value (RMSE: {rmse_value:.2f})')
plt.grid(True)
plt.tight_layout()
plt.savefig("images/error.png")
plt.close()