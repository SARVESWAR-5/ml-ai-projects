from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Dummy sequence data
X = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]])
y = np.array([4, 5, 6])

# Model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(3, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

# Predict
print("Prediction for [4,5,6]:", model.predict(np.array([[[4], [5], [6]]])))
