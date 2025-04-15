from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np

# Sample sequence data (X: a 3D array)
X = np.array([[[0], [1], [2]], [[1], [2], [3]], [[2], [3], [4]]])
y = np.array([3, 4, 5])

# Model
model = Sequential([
    SimpleRNN(10, activation='relu', input_shape=(3, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

# Predict
print("Prediction for [3,4,5]:", model.predict(np.array([[[3], [4], [5]]])))
