import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Load the dataset
url = "https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz"
data = pd.read_csv(url, sep=" ", header=None)

# Preprocess the dataset
X = data.iloc[:, 1:-1].values
y = data.iloc[:, 0].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Loop through different numbers of PCA components
n_components_list = list(range(1, 31))
ari_train_list = []
ari_test_list = []
acc_train_list = []
acc_test_list = []

for n_components in n_components_list:
    # Apply PCA with the current number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Apply K-means clustering to the reduced dataset
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans_labels_train = kmeans.fit_predict(X_train)
    kmeans_labels_test = kmeans.predict(X_test)

    # Evaluate the performance of K-means using the adjusted Rand index
    ari_train = adjusted_rand_score(y_train, kmeans_labels_train)
    ari_test = adjusted_rand_score(y_test, kmeans_labels_test)
    ari_train_list.append(ari_train)
    ari_test_list.append(ari_test)

    # Apply logistic regression to the reduced dataset
    logreg = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto', random_state=42)
    logreg.fit(X_train, y_train)
    y_pred_train = logreg.predict(X_train)
    y_pred_test = logreg.predict(X_test)

    # Evaluate the performance of logistic regression using accuracy
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train_list.append(acc_train)
    acc_test_list.append(acc_test)

# Plot the performance metrics for different numbers of PCA components
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.plot(n_components_list, ari_train_list, marker='o', color='b', label='ARI (K-means) - Train')
ax1.plot(n_components_list, ari_test_list, marker='o', color='b', linestyle='--', label='ARI (K-means) - Test')
ax2.plot(n_components_list, acc_train_list, marker='o', color='r', label='Accuracy (LogReg) - Train')
ax2.plot(n_components_list, acc_test_list, marker='o', color='r', linestyle='--', label='Accuracy (LogReg) - Test')

ax1.set_xlabel("Number of PCA Components")
ax1.set_ylabel("Adjusted Rand Index (K-means)")
ax2.set_ylabel("Accuracy (Logistic Regression)")
ax1.grid()

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
plt.savefig('class.png',pdi=500)