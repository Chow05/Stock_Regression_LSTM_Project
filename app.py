import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle

# Define your LSTM-based stock prediction model


class StockRegressionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_layers, dropout_prob):
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        super().__init__()
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# Load the trained model
def load_model(model_path, device, embedding_dim, hidden_size, n_layers, dropout_prob):
    model = StockRegressionModel(
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout_prob=dropout_prob
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Predict function
def predict(model, input_data, device):
    # Prepare input data for LSTM (1, 4)
    input_tensor = torch.tensor(
        input_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()


def main():
    st.title("LSTM-Based Stock Price Prediction")
    ticker_name = st.selectbox(
        "Choose a Ticker Name",
        ['PNJ', 'MSN', 'VIC', 'FPT']
    )

    # Input fields for stock features
    open_price = st.number_input("Open Price", min_value=0.0)
    high_price = st.number_input("High Price", min_value=0.0)
    low_price = st.number_input("Low Price", min_value=0.0)
    volume = st.number_input("Volume", min_value=0)

    if st.button("Predict"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Define hyperparameters
        embedding_dim = 1
        hidden_size = 8
        n_layers = 3
        dropout_prob = 0.15

        time_list = ['minute', 'hour', 'day']

        model_path_list = [f'best_models/{ticker_name}_minute_model.pt',
                           f'best_models/{ticker_name}_hour_model.pt',
                           f'best_models/{ticker_name}_day_model.pt']

        x_transform_path_list = [f'transformer/{ticker_name}_minute_x.pkl',
                                 f'transformer/{ticker_name}_hour_x.pkl',
                                 f'transformer/{ticker_name}_day_x.pkl']

        y_transform_path_list = [f'transformer/{ticker_name}_minute_y.pkl',
                                 f'transformer/{ticker_name}_hour_y.pkl',
                                 f'transformer/{ticker_name}_day_y.pkl']
        model_count = 0
        for model_path in model_path_list:
            print(model_path)
            model = load_model(model_path, device, embedding_dim,
                               hidden_size, n_layers, dropout_prob)

            with open(x_transform_path_list[model_count], 'rb') as xf:
                x_transform = pickle.load(xf)       # dict (mean, std)
            with open(y_transform_path_list[model_count], 'rb') as yf:
                y_transform = pickle.load(yf)       # dict (mean, std)

            trans_open_price = (
                open_price - x_transform['mean'][0]) / x_transform['std'][0]
            trans_high_price = (
                high_price - x_transform['mean'][1]) / x_transform['std'][1]
            trans_low_price = (
                low_price - x_transform['mean'][2]) / x_transform['std'][2]
            trans_volume_price = (
                volume - x_transform['mean'][3]) / x_transform['std'][3]
            trans_querry = np.array([[trans_open_price,
                                      trans_high_price,
                                      trans_low_price,
                                      trans_volume_price]])

            predicted_price = predict(model, trans_querry, device)
            predicted_price = (
                predicted_price * y_transform['std'][0]) + y_transform['mean'][0]

            volatility_price = predicted_price - open_price

            st.write(
                f"Volatility Stock Price ({time_list[model_count]}): ${volatility_price:.2f}")

            model_count += 1


if __name__ == "__main__":
    main()
