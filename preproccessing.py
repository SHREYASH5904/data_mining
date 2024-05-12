import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import resample

# Load the Iris dataset
iris_data = pd.read_csv('iris.csv')

# Display the first few rows of the dataset
print("Initial dataset:")
print(iris_data.head())

# Standardization
print("\nStandardization:")
scaler = StandardScaler()
standardized_data = scaler.fit_transform(iris_data.iloc[:, :-1])  # Standardize features, excluding the target variable
standardized_df = pd.DataFrame(standardized_data, columns=iris_data.columns[:-1])
print(standardized_df.head())

# Normalization
print("\nNormalization:")
min_max_scaler = MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(iris_data.iloc[:, :-1])  # Normalize features, excluding the target variable
normalized_df = pd.DataFrame(normalized_data, columns=iris_data.columns[:-1])
print(normalized_df.head())

# PCA Transformation
print("\nPCA Transformation:")
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(standardized_data)  # Apply PCA transformation to standardized data
pca_df = pd.DataFrame(data=pca_transformed, columns=['PC1', 'PC2'])
print(pca_df.head())

# Aggregation
print("\nAggregation:")
agg_data = iris_data.groupby('species').mean()  # Aggregate data by species and calculate the mean
print(agg_data)

# Discretization
print("\nDiscretization:")
discretized_data = pd.cut(iris_data['sepal_length'], bins=3, labels=['Short', 'Medium', 'Long'])  # Discretize sepal length into three bins
print(discretized_data.head())

# Binarization
print("\nBinarization:")
binarizer = preprocessing.Binarizer(threshold=iris_data['sepal_length'].mean())  # Binarize sepal length based on mean threshold
binarized_data = binarizer.fit_transform(iris_data[['sepal_length']])
print(binarized_data[:5])

# Sampling
print("\nSampling:")
sampled_data = resample(iris_data, n_samples=5)  # Randomly sample 5 instances from the dataset
print(sampled_data)
