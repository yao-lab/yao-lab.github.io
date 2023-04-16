




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
url = "https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz"
data = pd.read_csv(url, sep=" ", header=None)

# Preprocess the dataset
X = data.iloc[:, 1:-1].values
y = data.iloc[:, 0].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize the results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=50, edgecolors='black', linewidths=1, alpha=0.8)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.legend(*scatter.legend_elements(), title="Classes")
plt.savefig('fspca.png',dpi=500)
plt.show()

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Train a classifier using the reduced dataset
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:", confusion_mat)