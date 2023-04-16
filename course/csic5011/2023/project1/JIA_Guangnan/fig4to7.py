
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
data = np.loadtxt('zip.train')
y = data[:, 0]
X = data[:, 1:]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute the reconstruction error and reconstructed digit images for different numbers of principal components
#n_components_list = [10, 20, 30]
n_components_list = list(range(1, 31))
recon_errors = []
for n_components in n_components_list:
    # Apply PCA to the training data
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)

    # Transform the test data using the PCA model and compute the reconstruction error
    X_test_pca = pca.transform(X_test)
    X_test_recon = pca.inverse_transform(X_test_pca)
    #recon_error = mean_squared_error(X_test, X_test_recon)
    X_train_reduced = pca.transform(X_train)
    X_train_recon = pca.inverse_transform(X_train_reduced)
    recon_error = np.mean(np.square(X_train - X_train_recon))
    recon_errors.append(recon_error)

    
    # Display some original and reconstructed images
    n_images = 10
    fig, axs = plt.subplots(2, n_images, figsize=(10, 4))
    for i in range(n_images):
        axs[0, i].imshow(X_test[i].reshape(16, 16), cmap='gray')
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[0, i].set_title(f"Digit {y_test[i]}")
        axs[1, i].imshow(X_test_recon[i].reshape(16, 16), cmap='gray')
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        axs[1, i].set_title(f"Recon\nDigit {y_test[i]}")
    plt.suptitle(f"Number of Principal Components: {n_components}\nReconstruction Error: {recon_error:.2f}")
    plt.savefig(f"recon_{n_components}.png",pdi=500)
    plt.show()


plt.figure(2)
plt.plot(range(1,31),recon_errors)
plt.xlabel('Number of PCA Components')
plt.ylabel('Reconstruction Error')
plt.savefig('recon_error.png',pdi=500)
plt.show()

