import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd


#sample data
X = np.array([1, 2, 3, 4, 5]) #indep
y = np.array([2, 3, 4, 3, 5]) #dep

# Reshape X because it should be a 2D array for scikit-learn
X = X.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the data and the regression line
plt.scatter(X, y, label='Actual Data', color='blue')
plt.plot(X, y_pred, label='Regression Line', color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()

# Print the coefficients
print(f'Intercept (b0): {model.intercept_}')
print(f'Coefficient (b1): {model.coef_[0]}')