from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=5, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("MSE (Synthetic Data):", mean_squared_error(y_test, y_pred))
