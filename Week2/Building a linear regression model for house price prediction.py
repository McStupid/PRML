# Import package
# NumPy for numerical operations
# LinearRegression from scikit-learn for building a linear regression model.
import numpy as np
from sklearn.linear_model import LinearRegression
# Dataset, define input X and the corresponding target variable y using NumPy arrays
# Here X is data we collected (some features of houses), y is the house price
X = np.array([[1200, 2, 1, 1995],
[1500, 3, 2, 2002],
[1800, 3, 2, 1985],
[1350, 2, 1, 1998],
[2000, 4, 3, 2010]])
y = np.array([250, 320, 280, 300, 450])
# Create and train the model
# Create an instance of the LinearRegression model and fit it to the data (X,y)
model = LinearRegression()
model.fit(X, y)
# Prediction. After training, we can new use it to predict the prices for new houses
new_data = np.array([[1650, 3, 2, 2005],
[1400, 2, 1, 2000]])
predicted_prices = model.predict(new_data)
print("Predicted prices:", predicted_prices)