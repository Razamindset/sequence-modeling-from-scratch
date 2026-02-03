import numpy as np
import matplotlib.pyplot as plt
from linear_regression.with_adam import LinearRegressionAdam 
from sklearn.model_selection import train_test_split

# The idea is to predict the next vlaue int he sequnce given the past 10 20 vlaues 
# this can be either simple numbers say given 1-20 u predict the next one as 21 or next sine wave point

WINDOW_SIZE=20
DDATA_POINTS=1000
TRAIN_SPLIT = 0.8

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

data = gen_data(DDATA_POINTS)
X, y = gen_sequences(data, WINDOW_SIZE)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69, shuffle=False)


model = LinearRegressionAdam(lr=0.01, n_iters=500, batch_size=32, weight_decay=0.01)

print("----------Training Model----------")
model.fit(X, y, verbose=True)

predictions = model.predict(X_test)

# --- Visualization ---
plt.figure(figsize=(12, 6))

# actual data 
# plt.subplot(1, 2, 1)
plt.plot(y_test, label="Actual Data (Ground Truth)", color="black", alpha=0.3)
plt.plot(predictions, label="Model's prediction", color="red", linestyle="--")
plt.title(f"Ctual vs prediction")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()

plt.show()