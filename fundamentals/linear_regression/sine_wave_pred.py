import numpy as np
import matplotlib.pyplot as plt
from with_adam import LinearRegressionAdam 
from sklearn.model_selection import train_test_split

# The idea is to predict the next vlaue int he sequnce given the past 10 20 vlaues 
# this can be either simple numbers say given 1-20 u predict the next one as 21 or next sine wave point

#! By increasing WINDOW_SIZE and decresing weight decay we can significantly improve results
#! By Removing Noise in the data we can achieve a perect fit even in recursive operation
#! thirdly smoothing the data by moving average will also help with the noise

WINDOW_SIZE=50
DDATA_POINTS=1000
TRAIN_SPLIT = 0.8

# 200 steps becase after split only 200 data points are in the test dataset
FUTURE_STEPS=1000
START_POINT=0

def gen_data(n_points=1000):
    # 1000 points between 1 and 50
    time = np.linspace(0, 50, n_points)

    data = time
    #  we can eitehr use this or use a sine wave 
    # add some noise too
    data = np.sin(time) + np.random.normal(scale=0.05, size=n_points) 
    
    return data

def gen_sequences(data, window_size):
    """Create a sequneces of examples... previous tokens and the next one"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size])
    
    return np.array(X), np.array(y)

def recursive_predictions(model, initial_window, n_steps):
    """The idea is to start from a point and keep feeding the predicitons back in till STEPS and see what our model can produce... it is sine sequence modelling"""

    forecast = []
    current_window = initial_window.copy()

    for _ in range(n_steps):
        pred = model.predict(current_window.reshape(1, -1))[0]

        forecast.append(pred)

        current_window = np.roll(current_window, -1)
        current_window[-1] = pred
    
    return np.array(forecast)

def smooth_data(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='same')

data = gen_data(DDATA_POINTS)
data = smooth_data(data) # Clean the "distraction"
X, y = gen_sequences(data, WINDOW_SIZE)

# data = gen_data(DDATA_POINTS)
# X, y = gen_sequences(data, WINDOW_SIZE)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SPLIT, random_state=69, shuffle=False)


model = LinearRegressionAdam(lr=0.01, n_iters=500, batch_size=32, weight_decay=0)

print("----------Training Model----------")
model.fit(X, y, verbose=True)

print("----------Generating Future Recursive predictions----------")
forecast = recursive_predictions(model, initial_window=X_test[START_POINT], n_steps=FUTURE_STEPS)

# --- Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(y_test[START_POINT: FUTURE_STEPS], label="Actual Data (Ground Truth)", color="black", alpha=0.3)
plt.plot(forecast, label="Model's prediction", color="red", linestyle="--")
plt.title(f"Ctual vs prediction")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()

plt.show()

plt.figure(figsize=(10, 5))

# model.weights should be an array of size 50 (WINDOW_SIZE)
plt.bar(range(WINDOW_SIZE), model.w, color='royalblue', alpha=0.7)

plt.title("Model Weights: How much the model 'trusts' each lag")
plt.xlabel("Lag Index (0 is oldest, 19 is most recent)")
plt.ylabel("Weight Value")
plt.xticks(range(WINDOW_SIZE))
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()