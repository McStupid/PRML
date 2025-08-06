# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# Dataset, a simple one with five points
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
# Reshape X into a column vector
X = X.reshape(-1, 1)
# Define the degree of the polynomial, you can try other values
degree = 3
# Create polynomial features using PolynomialFeatures from the scikit-learn
# Transform X to polynomial features using fit_transform
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)
# Create and train a linear regression model
# Fit it to the polynomial features X_poly and the target variable y
model = LinearRegression()
model.fit(X_poly, y)
# Generate data for plotting
# Generate a range of X values using linspace, transform it to polynomial features
# Predict the corresponding y values using model.predict
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot = model.predict(X_plot_poly)
# Plot the original data and the fitted polynomial curve using matplotlib
plt.scatter(X, y, label='Original Data')
plt.plot(X_plot, y_plot, color='red', label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Fit of Degree {}'.format(degree))
plt.legend()
plt.show()