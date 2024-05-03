import json
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


# Load NLTK stopwords
stop_words = set(stopwords.words('english'))


# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    return processed_text


# Specify the path to your JSON file using a raw string literal
json_file_path = r"C:\Users\manue\Desktop\mental_health_survey\osmi-survey-2016_1479139902.json"

# Open the JSON file in read mode
with open(json_file_path, 'r') as file:
    # Load the JSON data into a Python object
    survey_data = json.load(file)

# Inspect the structure of survey_data
max_length = max(len(survey_data['questions']), len(survey_data['responses']))
print("Maximum length:", max_length)

# Pad or truncate each array to match the maximum length
survey_data['questions'] = survey_data['questions'][:max_length] + [None] * (max_length - len(survey_data['questions']))
survey_data['responses'] = survey_data['responses'][:max_length] + [None] * (max_length - len(survey_data['responses']))

# Convert the survey data to a list of dictionaries
records = []
for i in range(max_length):
    record = {'questions': survey_data['questions'][i], 'responses': survey_data['responses'][i]}
    records.append(record)

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(records)

# Convert any dictionaries in text fields to strings
text_fields = [col for col in df.columns if df[col].dtype == 'object']
df[text_fields] = df[text_fields].apply(lambda x: x.map(str) if x.dtype == 'object' else x)

# Impute missing numerical values with the mean
numerical_cols = df.select_dtypes(include=np.number).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Impute missing textual values with the mode
textual_cols = df.select_dtypes(include='object').columns
df[textual_cols] = df[textual_cols].fillna(df[textual_cols].mode().iloc[0])

# Check for missing values after imputation
missing_values = df.isnull().sum()

# Print the count of missing values for each column
print("Missing values after imputation:")
print(missing_values)

# Select a sample of textual data for testing
sample_text = df['responses'].iloc[0]  # Selecting the first row as an example

# Print the original text
print("Original Text:")
print(sample_text)
print()

# Preprocess the text
preprocessed_text = preprocess_text(sample_text)

# Print the preprocessed text
print("Preprocessed Text:")
print(preprocessed_text)

# Check for categorical features
categorical_cols = df.select_dtypes(include='object').columns
if not categorical_cols.empty:
    # Apply one-hot encoding to categorical features
    df = pd.get_dummies(df, columns=categorical_cols)

# Check for numerical features
numerical_cols = df.select_dtypes(include=np.number).columns
if not numerical_cols.empty:
    # Apply scaling to numerical features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Compute the covariance matrix
cov_X_std = np.cov(df.T)

# Compute the eigenvalues and eigenvectors of the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_X_std)
print("Eigenvalues:", eig_vals)

# Sort the eigenvalues in descending order
idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# Calculate the explained variance and cumulative explained variance
exp_var = eig_vals / np.sum(eig_vals) * 100
cum_exp_var = np.cumsum(exp_var)

# Print explained variance and cumulative explained variance
print("Explained variance:", exp_var)
print("Cumulative explained variance:", cum_exp_var)

# Project the original data onto the principal components
PR = eig_vecs[:, :2]
Y = df.dot(PR)

# Perform PCA using sklearn for comparison
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)

# Hierarchical Clustering
cluster = AgglomerativeClustering(n_clusters=5)
cluster.fit(X_pca)
labels = cluster.labels_

# Evaluate silhouette score
silhouette_avg = silhouette_score(X_pca, labels)
print("Silhouette Score:", silhouette_avg)

# Visualize clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Hierarchical Clustering')
plt.colorbar(label='Cluster')
plt.show()
