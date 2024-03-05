import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import requests
from datetime import datetime, timedelta
from keras.callbacks import EarlyStopping


def fetch_intraday_data(symbol, api_token, range='20Y'):
    url = f'https://cloud.iexapis.com/stable/stock/{symbol}/chart/{range}?token={api_token}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Handling missing values
        df.fillna(method='ffill', inplace=True)

        # Exclude non-numeric columns
        numeric_data = df.select_dtypes(include=[np.number])

        return numeric_data
    except requests.exceptions.RequestException as e:
        print("Error fetching data:", e)
        return None

def fetch_sp500_data(api_token, range='20Y'):
    url = f'https://cloud.iexapis.com/stable/stock/market/index/spy/chart/{range}?token={api_token}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        print("Error fetching S&P 500 data:", e)
        return None

def train_lstm_model(data, time_steps=None, epochs=20, validation_split=0.2):
    # Split data into training and validation sets
    train_size = int(len(data) * (1 - validation_split))
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    # Create early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Normalize the data
    scaler = MinMaxScaler()  # Create scaler instance
    scaled_train_data = scaler.fit_transform(train_data)  # Fit on training set
    scaled_val_data = scaler.transform(val_data)  # Apply to validation set

    # Prepare training data
    X_train, y_train = [], []
    for i in range(len(scaled_train_data) - time_steps):
        X_train.append(scaled_train_data[i: (i + time_steps), :])
        y_train.append(scaled_train_data[i + time_steps, :])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Prepare validation data
    X_val, y_val = [], []
    for i in range(len(scaled_val_data) - time_steps):
        X_val.append(scaled_val_data[i: (i + time_steps), :])
        y_val.append(scaled_val_data[i + time_steps, :])

    X_val, y_val = np.array(X_val), np.array(y_val)

    # Create the LSTM network
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
    model.add(LSTM(units=32, return_sequences=True))  # Second LSTM layer
    model.add(LSTM(units=32))  # Third LSTM layer
    model.add(Dense(X_train.shape[2]))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model with early stopping
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
                        validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Make predictions on the validation set
    val_predictions = model.predict(X_val)

    # Inverse transform to get original prices
    val_predictions = scaler.inverse_transform(val_predictions)
    y_val_orig = scaler.inverse_transform(y_val)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_val_orig, val_predictions)
    mse = mean_squared_error(y_val_orig, val_predictions)

    print("Mean Absolute Error on Validation Set:", mae)
    print("Mean Squared Error on Validation Set:", mse)

    return model, scaler

def get_predicted_price(model, scaler, time_steps, data, spy_data):
    # Merge TSLA and SPY data
    merged_data = pd.merge(data, spy_data, left_index=True, right_index=True, suffixes=('_TSLA', '_SPY'))

    # Apply data normalization
    scaled_data = scaler.transform(merged_data.astype(float))

    # Prepare data for LSTM
    X = []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i: (i + time_steps), :])
    X = np.array(X)

    # Make predictions
    x_pred = scaled_data[-time_steps:]
    x_pred = np.reshape(x_pred, (1, x_pred.shape[0], x_pred.shape[1]))
    predicted_price = model.predict(x_pred)

    # Inverse transform to get original prices
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price

# IEX Cloud API token with stock ticker
api_token = 'pk_de0a3502f30445c9adf9ebb0440afbda'
symbol = 'TSLA'

# Fetch SPY data
spy_data = fetch_sp500_data(api_token, range='20Y')

# Fetch intraday data for the past 20 years
data = fetch_intraday_data(symbol, api_token, range='20Y')

if data is not None:
    print("Intraday data:")
    print(data.head())

    # Train multiple LSTM models with bagging
    num_models = 10
    models = []
    scalers = []
    for _ in range(num_models):
        model, scaler = train_lstm_model(data, time_steps=100)
        models.append(model)
        scalers.append(scaler)

    # Get predictions from each model
    predicted_prices = []
    for model, scaler in zip(models, scalers):
        predicted_price = get_predicted_price(model, scaler, 100, data, spy_data)
        predicted_prices.append(predicted_price)

    # Use median instead of average for more robust prediction
    median_predicted_price = np.median(predicted_prices, axis=0)

    # Assume current price of stock (you can get this from your data or another source)
    current_price = data.iloc[-1]['close']

    # Calculate the expected price based on the average predicted change
    predicted_change = median_predicted_price[0][0]
    expected_price = current_price + predicted_change
    print("Expected price:", expected_price)

    # Convert predicted_change to integer before using it in timedelta calculation
    predicted_change_days = int(predicted_change)

    # List to store forecasted dates and expected prices
    forecasts = []

    # Calculate the forecasted date and expected price for each specified time horizon
    forecasted_dates = []
    forecasted_prices = []

    # Current date
    current_date = data.index[-1]

    # Time horizons in days
    time_horizons = [7, 14, 30, 60, 90]

    for horizon in time_horizons:
        forecasted_date = current_date + timedelta(days=horizon)
        forecasted_price = current_price + horizon * predicted_change
        forecasted_dates.append(forecasted_date)
        forecasted_prices.append(forecasted_price)

    # Print forecasted dates and expected prices
    for i in range(len(forecasted_dates)):
        print("Forecasted date for {} days:".format(time_horizons[i]))
        print("Date:", forecasted_dates[i])
        print("Expected price:", forecasted_prices[i])

    # Compare the ensemble model's predicted price change with the current price
    if predicted_change > 0:
        print("The stock is expected to rise.")
    elif predicted_change < 0:
        print("The stock is expected to fall.")
    else:
        print("The stock is expected to remain stable.")
