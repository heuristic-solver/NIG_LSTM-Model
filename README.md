<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stock Price Prediction using FFT and Machine Learning Models</title>
</head>
<body>

<h1 style="text-align: center;">STOCK PRICE PREDICTION USING FFT AND MACHINE LEARNING MODELS</h1>

<p>This repository contains code for a stock price prediction project. It leverages Fourier Transform filtering 
to process stock price data, followed by machine learning models like XGBoost and neural networks to make predictions. 
Key metrics such as Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE), and R-squared (R²) are 
computed to evaluate model performance.</p>

<h2 style="text-align: center;">FEATURES</h2>

<ul>
  <li>Fourier Transform Filtering: Applies FFT to remove noise from the stock prices, capturing only significant frequency components.</li>
  <li>Machine Learning Models:
    <ul>
      <li>XGBoost for regression on filtered stock data.</li>
      <li>Neural Network with custom loss functions to predict prices and confidence intervals.</li>
    </ul>
  </li>
  <li>Metrics for Evaluation: Includes MAPE, MSE, and R².</li>
  <li>Data Visualization: Plots of actual vs. predicted prices and residual analysis.</li>
</ul>

<h2 style="text-align: center;">DATASET</h2>

<p>The code utilizes the <code>Apple_stock_yahoo_closehigh.csv</code> dataset, which includes:</p>
<ul>
  <li>Date</li>
  <li>Close (Stock closing price)</li>
</ul>

<h2 style="text-align: center;">INSTALLATION</h2>

<ol>
  <li>Clone this repository:
    <pre><code>git clone https://github.com/heuristic-solver/NIG_LSTM-Model.git
cd NIG_LSTM-Model</code></pre>
  </li>
</ol>

<h2 style="text-align: center;">USAGE</h2>

<h3 style="text-align: center;">Data Preprocessing and FFT Filtering</h3>

<p>The code reads and processes the stock data by applying FFT:</p>
<pre><code>import numpy as np
import pandas as pd

data = pd.read_csv('Apple_stock_yahoo_closehigh.csv')
close_fft = np.fft.fft(np.asarray(data['Close'].tolist()))
# Set insignificant frequency components to zero
fft_list = np.asarray(close_fft)
fft_list[500:-500] = 0
filtered_close = np.fft.ifft(fft_list)
data['Filtered_Close'] = filtered_close.real
</code></pre>

<h3 style="text-align: center;">Model Training</h3>

<p><strong>XGBoost Model:</strong> After filtering, data is split and trained using XGBoost:</p>
<pre><code>from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = data[['Filtered_Close']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

xgb_model = XGBRegressor(n_estimators=300, random_state=42)
xgb_model.fit(X_train, y_train)
</code></pre>

<p><strong>Neural Network Model:</strong> A neural network model is implemented with a custom loss function to predict stock prices along with confidence intervals:</p>
<pre><code>from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras

model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(optimizer='adam', loss=weighted_mse)
</code></pre>

<h2 style="text-align: center;">Evaluation Metrics</h2>

<p>After training, the following metrics are used to evaluate the models:</p>
<ul>
  <li>Mean Squared Error (MSE)</li>
  <li>Mean Absolute Percentage Error (MAPE)</li>
  <li>R-squared (R²)</li>
</ul>

<p>Example:</p>
<pre><code>from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}, R2: {r2}")
</code></pre>

<h2 style="text-align: center;">Visualization</h2>

<p>Plots of actual vs. predicted prices and residual analysis for model diagnostics.</p>
<pre><code>import matplotlib.pyplot as plt

plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.legend()
plt.show()
</code></pre>

<h2 style="text-align: center;">LICENSE</h2>

<p>This project is licensed under the MIT License.</p>

<h2 style="text-align: center;">CONTACT</h2>

<p>For any questions, please contact <a href="https://github.com/heuristic-solver">heuristic-solver</a>.</p>

</body>
</html>
