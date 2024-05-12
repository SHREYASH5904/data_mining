import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset
wine_data = pd.read_csv('wine_quality.csv')

# Display the first few rows of the dataset
print("Initial dataset:")
print(wine_data.head())

# Handling Missing Values
print("\nHandling Missing Values:")
print("Number of missing values per column:")
print(wine_data.isnull().sum())

# Fill missing values with mean or median
wine_data.fillna(wine_data.mean(), inplace=True)

# Handling Outliers
print("\nHandling Outliers:")
z_scores = stats.zscore(wine_data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
wine_data = wine_data[filtered_entries]

# Handling Inconsistent Values
print("\nHandling Inconsistent Values:")
# Let's assume alcohol content should be between 0 and 20
wine_data = wine_data[(wine_data['alcohol'] >= 0) & (wine_data['alcohol'] <= 20)]

# Define validation rules based on domain knowledge or dataset characteristics
# For example, alcohol content should not be negative, pH should be between 0 and 14, etc.

# Validate the dataset based on defined rules
print("\nPerforming Validations:")
# Check if alcohol content is within a valid range
invalid_alcohol = wine_data[(wine_data['alcohol'] < 0) | (wine_data['alcohol'] > 20)]
print("Invalid alcohol content entries:")
print(invalid_alcohol)

# Check if pH is within a valid range
invalid_pH = wine_data[(wine_data['pH'] < 0) | (wine_data['pH'] > 14)]
print("\nInvalid pH entries:")
print(invalid_pH)

