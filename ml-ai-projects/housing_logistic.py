from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load and transform data into classification task
data = fetch_california_housing()
y = (data.target > data.target.mean()).astype(int)  # High price = 1, Low = 0
X_train, X_test, y_train, y_test = train_test_split(data.data, y, test_size=0.2)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
