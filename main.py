# Import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import requests

def fetch_intraday_data(symbol, api_token, range='5dm'):
    url = f'https://cloud.iexapis.com/stable/stock/{symbol}/chart/{range}?token={api_token}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.drop(columns=['minute'], inplace=True)  # Drop non-numeric column 'minute'
        return df
    except requests.exceptions.RequestException as e:
        print("Error fetching data:", e)
        return None

def train_lstm_model(data, time_steps=100, epochs=10):
    # Exclude non-numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    if not numeric_data.empty:
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(numeric_data.values)

        # Prepare the data for LSTM
        if len(scaled_data) > time_steps:
            X = []  # Initialize X as an empty list
            for i in range(len(scaled_data) - time_steps):
                X.append(scaled_data[i: (i + time_steps), :])  # Append directly as NumPy arrays

            X = np.array(X)  # Convert the list of windows to a NumPy array
            y = scaled_data[time_steps:, :]  # Create the target values

            # Get the number of features from the shape of the windows in X
            num_features = X.shape[2]

            # Create the LSTM network
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, num_features)))
            model.add(LSTM(units=50))
            model.add(Dense(num_features))  # Output layer with the same number of features as input

            model.compile(loss='mean_squared_error', optimizer='adam')

            # Train the model
            model.fit(X, y, epochs=epochs, batch_size=1, verbose=2)

            return model, scaler
        else:
            print("Not enough data for the specified time steps.")
    else:
        print("No numeric columns found in the data.")
    return None, None

def get_predicted_price(model, scaler, data):
    # Separate numerical features (excluding time)
    numeric_data = data[:, 1:].astype(float)  # Assuming time is in the first column, convert to float

    # Check if all values are numerical
    if np.isnan(numeric_data).any():
        # Handle NaN values (e.g., impute or remove)
        numeric_data = numeric_data[~np.isnan(numeric_data).any(axis=1)]  # Remove rows containing NaN values

    # Apply data normalization
    scaled_data = scaler.transform(numeric_data)

    # Prepare data for LSTM
    time_steps = 100  # Example value, adjust as needed
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

# Define a threshold value
threshold = 5  # You can adjust this threshold value according to preference

# IEX Cloud API token with stock ticker
api_token = 'pk_91452d2ecfaa41aead503d79464f6005'
symbol = 'NVDA'
data = fetch_intraday_data(symbol, api_token)

if data is not None:
    print("Intraday data:")
    print(data.head())

    # Train LSTM model
    model, scaler = train_lstm_model(data)

    # Assume current price of stock (you can get this from your data or another source)
    current_price = data.iloc[-1]['close']

    # Get the predicted price
    if model is not None and scaler is not None:
        predicted_price = get_predicted_price(model, scaler, data.values)
        print("Predicted price:", predicted_price)

        # Calculate the expected price based on the predicted change
        predicted_change = predicted_price[0][0]
        expected_price = current_price + predicted_change
        print("Expected price:", expected_price)

        # Compare the LSTM model's predicted price change with the current price
        if predicted_change > 0:
            print("The stock is expected to rise.")
        elif predicted_change < 0:
            print("The stock is expected to fall.")
        else:
            print("The stock is expected to remain stable.")
