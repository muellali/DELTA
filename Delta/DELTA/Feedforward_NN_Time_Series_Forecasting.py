
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Step 1: Data Simulation
np.random.seed(0)
data_points = 365  # For example, daily data for one year
time_series_data = np.sin(2 * np.pi * np.arange(data_points) / 30) + np.random.normal(0, 0.5, data_points)
dates = pd.date_range(start='2023-01-01', periods=data_points, freq='D')
time_series_df = pd.DataFrame(data={'Date': dates, 'Value': time_series_data})
time_series_df.set_index('Date', inplace=True)

# Plotting the simulated data
plt.figure(figsize=(10, 5))
plt.plot(time_series_df.index, time_series_df['Value'])
plt.title('Simulated Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Step 2: Data Preparation
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series_df)

# Function to create dataset for training
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 1
X, Y = create_dataset(scaled_data, look_back)

# Splitting data into training and testing sets
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
trainX, testX = X[0:train_size], X[train_size:len(X)]
trainY, testY = Y[0:train_size], Y[train_size:len(Y)]

# Step 3: Model Building
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Step 4: Model Training
model.fit(trainX, trainY, epochs=100, batch_size=2, verbose=2)

# Step 5: Forecasting
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inverse scaling for plotting
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Plotting baseline and predictions
plt.figure(figsize=(10, 5))
plt.plot(time_series_df.index, scaler.inverse_transform(scaled_data), label='Original data')
plt.plot(time_series_df.index[look_back:len(trainPredict)+look_back], trainPredict, label='Training predictions')
plt.plot(time_series_df.index[len(trainPredict)+(look_back*2)+1:len(scaled_data)-1], testPredict, label='Test predictions')
plt.title('Forecasting with Feedforward Neural Network')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Evaluation
testScore = mean_squared_error(testY[0], testPredict[:,0])
print('Test MSE:', testScore)
