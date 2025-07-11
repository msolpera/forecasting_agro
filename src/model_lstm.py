import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import pandas as pd

class LSTMPipeline:
    def __init__(self, epochs=50, batch_size=16, metric='rmse'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.metric = metric
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None


    def fit(self, train_feat, target_col):

        y_train_scaled = self.scaler_y.fit_transform(train_feat[[target_col]])
        X_train_scaled = self.scaler_x.fit_transform(train_feat.drop(columns=target_col))
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

        model = Sequential()
        model.add(LSTM(50, activation='tanh', input_shape=(1, X_train_scaled.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_scaled, y_train_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        self.model = model

        # Guardar predicciones en train
        pred_scaled = self.model.predict(X_train_scaled)
        self.y_pred_train = self.scaler_y.inverse_transform(pred_scaled).flatten()
        self.y_train_true = train_feat[target_col]

        return self

    def predict(self, test_feat, target_col):
        self.y_test_true = test_feat[target_col]

        X_test_scaled = self.scaler_x.transform(test_feat.drop(columns=target_col))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        pred_scaled = self.model.predict(X_test_scaled)
        self.y_pred_test = self.scaler_y.inverse_transform(pred_scaled).flatten()

        return self.y_pred_test

    def score(self, y_true, y_pred, metric=None):
        metric = metric or self.metric
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        elif metric == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError("Use: 'rmse', 'mae', 'mape', 'r2'.")

    def report(self):
        if not hasattr(self, 'y_pred_test'):
            raise RuntimeError("Debés correr `.predict()` antes de usar `.report()`.")

        print(f"MAE_test: {mean_absolute_error(self.y_test_true, self.y_pred_test):.2f}")
        print(f"RMSE_test: {np.sqrt(mean_squared_error(self.y_test_true, self.y_pred_test)):.2f}")
        print(f"MAPE_test: {np.mean(np.abs((self.y_test_true - self.y_pred_test) / self.y_test_true)) * 100:.2f}%")

        print('----')
        print(f"MAE_train: {mean_absolute_error(self.y_train_true, self.y_pred_train):.2f}")
        print(f"RMSE_train: {np.sqrt(mean_squared_error(self.y_train_true, self.y_pred_train)):.2f}")
        print(f"MAPE_train: {np.mean(np.abs((self.y_train_true - self.y_pred_train) / self.y_train_true)) * 100:.2f}%")

    def plot_predictions(self, title="LSTM Forecast vs Real"):
        if not hasattr(self, 'y_pred_test'):
            raise RuntimeError("Run first`.predict()`.")
        plt.figure(figsize=(10, 6))
        plt.plot(self.y_train_true.index, self.y_train_true, label='Train')
        plt.plot(self.y_test_true.index, self.y_test_true, label='Test Real', color='red')
        plt.plot(self.y_test_true.index, self.y_pred_test, label='Test Predicted', color='blue')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

