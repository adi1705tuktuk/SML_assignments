import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot

# Load the dataset
data = np.load('/Users/adityaprasad/Desktop/SML_A2_2022036/mnist.npz')

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# Define the reshape function
def reshape(X, p, eig_vectors):
    Up = eig_vectors[:, :p]
    Yp = np.dot(Up.T, X)
    return np.dot(Up, Yp)

# Define a function to perform dimensionality reduction and reconstruction
def perform_dimensionality_reduction(class_collection, p, U, mean_collection, X_nomean):
    X_trial_list = np.zeros((50, 784))
    X_recon_list = np.zeros((50, 784))
    for i in range(10):
        for j in range(5):
            X_trial_list[5*i+j] = (class_collection.T)[100*i+j]
            X_recon_list[5*i+j] = (reshape(X_nomean[:, 100*i+j], p, U)).T + mean_collection

    X_trial_list = X_trial_list.reshape(50, 28, 28)
    X_recon_list = X_recon_list.reshape(50, 28, 28)
    return X_trial_list, X_recon_list

def mean_cov_calc(X, U, p, N, y_train):
    mean_collection = np.zeros((10, p))
    covariance_collection = np.zeros((10, p, p))
    lamb = (1e-2)
    mod_matrix = lamb * np.identity(p)
    for i in range(10):
        condition = y_train == i
        vector_train = (X.T)[condition]
        U_vect = U[:, :p]
        Y = multi_dot([U_vect.T, vector_train.T])
        mean_collection[i] = np.mean(Y, axis=1)
        covariance_collection[i] = (np.cov(Y, bias=True) + mod_matrix)
    return mean_collection, covariance_collection

def QDA(V, K, p, mean_collection, covariance_collection, N):
    quadratic_forms = np.zeros((10, N))
    for i in range(10):
        U_vect = K[:, :p]
        Y = multi_dot([U_vect.T, V])
        U = mean_collection[i]
        C = covariance_collection[i]
        C_inv = np.linalg.inv(C)
        log_det = -0.5 * np.linalg.slogdet(C)[1]
        for j in range(N):
            X = Y[:, j]
            term = -0.5 * multi_dot([(X - U).T, C_inv, (X - U)]) + log_det
            quadratic_forms[i][j] = term
    return quadratic_forms

def accuracy(Q, N, y_test, p):
    max_index = np.argmax(Q, axis=0)
    correct = np.sum(max_index == y_test)
    accuracy = correct / N
    print(f"Accuracy for p={p}: {accuracy}")
    return accuracy

# Flatten the training and testing data
vectorized_train = x_train.reshape(x_train.shape[0], -1)
vectorized_test = x_test.reshape(x_test.shape[0], -1)

# Calculate mean and covariance for training data
mean_terms_train = np.mean(vectorized_train.T, axis=1)
X_nomean_train = vectorized_train.T - np.tile(mean_terms_train, (vectorized_train.shape[0], 1)).T
S_train = multi_dot([X_nomean_train, X_nomean_train.T]) / (len(X_nomean_train[0]) - 1)

eig_values_train, eig_vectors_train = np.linalg.eig(S_train)

sorted_indices = np.argsort(eig_values_train)[::-1]
eig_vsort_train = eig_values_train[sorted_indices]
U_train = eig_vectors_train[:, sorted_indices]

# Calculate mean and covariance for testing data
mean_terms_test = np.mean(vectorized_test.T, axis=1)
X_nomean_test = vectorized_test.T - np.tile(mean_terms_test, (vectorized_test.shape[0], 1)).T
S_test = multi_dot([X_nomean_test, X_nomean_test.T]) / (len(X_nomean_test[0]) - 1)

eig_values_test, eig_vectors_test = np.linalg.eig(S_test)

sorted_indices = np.argsort(eig_values_test)[::-1]
eig_vsort_test = eig_values_test[sorted_indices]
U_test = eig_vectors_test[:, sorted_indices]

# Define class collection from training data
class_collection = np.array([vectorized_train[y_train == i][:100] for i in range(10)]).reshape(1000, 784).T

# Specify different values of p
p_values = [5, 10, 20]

# Perform dimensionality reduction and reconstruction for each value of p
for p in p_values:
    X_trial_list, X_recon_list = perform_dimensionality_reduction(class_collection, p, U_train, mean_terms_train, X_nomean_train)
    
    # Calculate Mean Squared Error (MSE)
    MSE = np.mean((X_recon_list - X_trial_list) ** 2)
    
    # Print MSE
    print(f"MSE for p={p}: {MSE}")
    
    # Perform QDA and calculate accuracy for each value of p
    mean_collection, covariance_collection = mean_cov_calc(X_nomean_train, U_train, p, vectorized_train.shape[0], y_train)
    QDA_terms = QDA(X_nomean_test, U_train, p, mean_collection, covariance_collection, vectorized_test.shape[0])
    accuracy(QDA_terms, vectorized_test.shape[0], y_test, p)

    # Display the reconstructed images
    for i in range(50):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(X_trial_list[i].reshape(28, 28), cmap='gray')
        axes[0].set_title("Original")
        axes[1].imshow(X_recon_list[i].reshape(28, 28), cmap='gray')
        axes[1].set_title("Reconstructed")
        for ax in axes:
            ax.axis('off')
        plt.show()
