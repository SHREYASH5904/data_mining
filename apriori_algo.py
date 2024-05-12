import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Display the first few rows of the dataset
print("Titanic dataset:")
print(titanic_data.head())

# Data preprocessing: Select relevant columns
titanic_data = titanic_data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Convert categorical variables to binary format
titanic_data_binary = pd.get_dummies(titanic_data, columns=['Pclass', 'Sex'])

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets_titanic = 0


# Print frequent itemsets
print("\nFrequent Itemsets (Titanic):")
print(frequent_itemsets_titanic)

# Generate association rules
association_rules_titanic = association_rules(frequent_itemsets_titanic, metric="confidence", min_threshold=0.5)

# Print association rules
print("\nAssociation Rules (Titanic):")
print(association_rules_titanic)

# Load the Black Friday dataset
black_friday_data = pd.read_csv('BlackFriday.csv')

# Display the first few rows of the dataset
print("\nBlack Friday dataset:")
print(black_friday_data.head())

# Data preprocessing: Select relevant columns
black_friday_data = black_friday_data[['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Purchase']]

# Convert categorical variables to binary format
black_friday_data_binary = pd.get_dummies(black_friday_data, columns=['Gender', 'Age', 'City_Category'])

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets_black_friday = apriori(black_friday_data_binary, min_support=0.001, use_colnames=True)

# Print frequent itemsets
print("\nFrequent Itemsets (Black Friday):")
print(frequent_itemsets_black_friday)

# Generate association rules
association_rules_black_friday = association_rules(frequent_itemsets_black_friday, metric="confidence", min_threshold=0.5)

# Print association rules
print("\nAssociation Rules (Black Friday):")
print(association_rules_black_friday)
