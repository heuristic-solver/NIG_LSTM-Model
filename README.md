                        STOCK PRICE PREDICTION USING FFT AND MACHINE LEARNING MODELS

This repository contains code for a stock price prediction project. It leverages Fourier Transform filtering 
to process stock price data, followed by machine learning models like XGBoost and neural networks to make predictions. 
Key metrics such as Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE), and R-squared (R²) are 
computed to evaluate model performance.


                                    FEATURES

- Fourier Transform Filtering: Applies FFT to remove noise from the stock prices, capturing only significant frequency components.
- Machine Learning Models:
  - XGBoost for regression on filtered stock data.
  - Neural Network with custom loss functions to predict prices and confidence intervals.
- Metrics for Evaluation: Includes MAPE, MSE, and R².
- Data Visualization: Plots of actual vs. predicted prices and residual analysis.


                                    DATASET

The code utilizes the `Apple_stock_yahoo_closehigh.csv` dataset, which includes:
- Date
- Close (Stock closing price)


                                    INSTALLATION

1. Clone this repository:
    git clone https://github.com/yourusername/yourrepo.git
    cd yourrepo
2. Install the required packages:
    pip install -r requirements.txt


                                    USAGE

                            Data Preprocessing and FFT Filtering

The code reads and processes the stock data by applying FFT:

import numpy as np
import pandas as pd

data = pd.read_csv('Apple_stock_yahoo_closehigh.csv')
close_fft = np.fft.fft(np.asarray(data['Close'].tolist()))
# Set insignificant frequency components to zero
fft_list = np.asarray(close_fft)
fft_list[500:-500] = 0
filtered_close = np.fft.ifft(fft_list)
data['Filtered_Close'] = filtered_close.real


                            Model Training

1. XGBoost Model: After filtering, data is split and trained using XGBoost:

   from xgboost import XGBRegressor
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import MinMaxScaler

   X = data[['Filtered_Close']]
   y = data['Close']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
   xgb_model = XGBRegressor(n_estimators=300, random_state=42)
   xgb_model.fit(X_train, y_train)

2. Neural Network Model: A neural network model is implemented with a custom loss function to predict stock prices along with confidence intervals:

   from keras.models import Sequential
   from keras.layers import Dense
   from tensorflow import keras

   model = Sequential()
   model.add(Dense(64, activation='relu'))
   model.add(Dense(64, activation='relu'))
   model.add(Dense(2, activation='linear'))
   model.compile(optimizer='adam', loss=weighted_mse)


                            Evaluation Metrics

After training, the following metrics are used to evaluate the models:
- Mean Squared Error (MSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²)

Example:

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}, R2: {r2}")


                            Visualization

- Plots of actual vs. predicted prices
- Residuals plot for model diagnostics

Example:

import matplotlib.pyplot as plt

plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.legend()
plt.show()


                                    LICENSE

This project is licensed under the MIT License.


                                    CONTACT

For any questions, please contact [Your Name](https://github.com/yourusername).
