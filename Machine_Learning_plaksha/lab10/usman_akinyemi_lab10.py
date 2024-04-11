"""
Lab 10 Machine Learning Assignment
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#=========================================================================================
np.random.seed(0)
mu1 = [2, 2]
sigma1 = [[0.9, -0.0255], [-0.0255, 0.9]]
dist1 = np.random.multivariate_normal(mu1, sigma1, 250)

mu2 = [5, 5]
sigma2 = [[0.5, 0], [0, 0.3]]
dist2 = np.random.multivariate_normal(mu2, sigma2, 250)
#=========================================================================================
X = np.vstack((dist1, dist2))
y = np.hstack((np.zeros(250), np.ones(250)))
#=========================================================================================
X = np.hstack((np.ones((X.shape[0], 1)), X))
#=========================================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#=========================================================================================
#=========================================================================================
def step_activation(z):
    return 1 if z >= 0 else 0

def perceptron_train(X, y, learning_rate, max_epochs):
    weights = np.zeros(X.shape[1])
    for _ in range(max_epochs):
        for i in range(X.shape[0]):
            z = np.dot(X[i], weights)
            y_pred = step_activation(z)
            weights += learning_rate * (y[i] - y_pred) * X[i]
    return weights
#=========================================================================================
learning_rate = 0.1
max_epochs = 1000
weights = perceptron_train(X_train, y_train, learning_rate, max_epochs)
#=========================================================================================
x_values = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
y_values = -(weights[0] + weights[1] * x_values) / weights[2]

plt.figure(figsize=(8, 6))
plt.scatter(dist1[:, 0], dist1[:, 1], color='blue', label='Class 0')
plt.scatter(dist2[:, 0], dist2[:, 1], color='red', label='Class 1')
plt.plot(x_values, y_values, color='green', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary for two classes')
plt.legend()
plt.show()
#=========================================================================================
def predict(X, weights):
    return np.array([step_activation(np.dot(x, weights)) for x in X])

y_pred = predict(X_test, weights)
conf_mat = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_mat)
#=========================================================================================
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
