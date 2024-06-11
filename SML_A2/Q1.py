import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot

# Load the dataset
data = np.load('/Users/adityaprasad/Desktop/SML_A2_2022036/mnist.npz')

x_train=data['x_train']
y_train=data['y_train']
x_test=data['x_test']
y_test=data['y_test']

# Function to display samples
def display_samples(samples):
    for j, sample in enumerate(samples):
        plt.imshow(sample, cmap='gray')
        plt.title(f"Sample {j+1}")
        plt.axis('off')
        plt.show()

# Function to calculate accuracy
def calculate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels)

# Initialize a list to store 5 samples for each class
class_samples = [[] for _ in range(10)]

# Collect 5 samples for each class
for i in range(10):
    condition = (y_train == i)
    samples = x_train[condition][:5]
    class_samples[i] = samples

# Display samples
for i, samples in enumerate(class_samples):
    print("Class", i)
    display_samples(samples)
    print("\n")

# Training: Calculate mean and covariance for each class
mean_collection = np.zeros((10, 784))
covariance_collection = np.zeros((10,784,784))
lamb = 1e-2
mod_matrix = lamb * np.identity(784)
vectorized_train = x_train.flatten().reshape(x_train.shape[0], 28*28)

for i in range(10):
    condition = y_train == i
    v_train = vectorized_train[condition].T
    mean_collection[i] = np.mean(v_train, axis=1)
    covariance_collection[i] = np.cov(v_train, bias=True) + mod_matrix

# Testing: Calculate quadratic forms and predict
vectorized_test = x_test.flatten().reshape(x_test.shape[0], 28*28)
quadratic_forms = np.zeros((10, x_test.shape[0]))

for i in range(10):
    U = mean_collection[i]
    C = covariance_collection[i]
    C_inv = np.linalg.inv(C)
    log_det = -0.5 * np.linalg.slogdet(C)[1]
    for j in range(vectorized_test.shape[0]):
        X = vectorized_test[j]
        term = -0.5 * np.dot(np.dot((X - U).T, C_inv), (X - U)) + log_det
        quadratic_forms[i][j] = term

# Prediction
max_index = np.argmax(quadratic_forms, axis=0)

# Calculate accuracy
accuracy = calculate_accuracy(max_index, y_test)
print("Accuracy:", accuracy)
