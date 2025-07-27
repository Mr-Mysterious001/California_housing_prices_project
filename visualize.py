import pandas as pd
import numpy as np
from main import build_pipeline
import matplotlib.pyplot as plt 


data = pd.read_csv("housing.csv")
num_attribs = data.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]
pipe = build_pipeline(num_attribs, cat_attribs)
pipe.fit_transform(data)

#Visualizing the data using graphs in matplotlib

features = ['median_income', 'housing_median_age', 'population', 'median_house_value']
pd.plotting.scatter_matrix(data[features],c=data['median_house_value'], cmap='viridis', alpha=0.5, figsize=(8, 8), diagonal='hist')
plt.savefig("scatter_matrix.png")
plt.close()


plt.figure()
plt.scatter(data['longitude'], data['latitude'], c=data['median_house_value'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Median House Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Distribution of Median House Value')
plt.savefig("Lat_Long.png")  # give meaningful names if multiple plots
plt.close()

