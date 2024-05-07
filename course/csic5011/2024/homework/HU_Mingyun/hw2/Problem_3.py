import numpy as np
import matplotlib.pyplot as plt

# Set parameters
n = 400    # matrix size
num_samples = 500    # number of random matrices to generate

# Generate random matrices from GOE
A = np.random.randn(n, n)
A = (A + A.T)/2    # make symmetric
A /= np.sqrt(n)    # rescale by 1/sqrt(n)

# Compute eigenvalues of each matrix and stack them into a vector
eigenvalues = np.zeros(n*num_samples)
for i in range(num_samples):
    eigenvalues[i*n:(i+1)*n] = np.linalg.eigvalsh(A)

# Compute empirical spectral density (ESD) of eigenvalues
bins = np.linspace(-2, 2, 100)
hist, _ = np.histogram(eigenvalues, bins=bins, density=True)
esd = np.cumsum(hist)*(bins[1] - bins[0])

# Compute theoretical semi-circle distribution
r = np.sqrt(4 - bins**2)
semi_circle = 1/(2*np.pi)*np.maximum(r, 0)

# Plot ESD and theoretical semi-circle distribution
plt.plot(bins, esd, label='ESD')
plt.plot(bins, semi_circle, label='Semi-circle')
plt.legend()
plt.show()