import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class ARIMAPipeline:
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), log=True):
        self.order = order
        self.seasonal_order = seasonal_order
        self.log = log
        self.model = None
        self.result = None

    def fit(self, y_train, exog_train=None):
        self.y_train = np.log(y_train) if self.log else y_train
        self.exog_train = exog_train

        self.model = SARIMAX(
            self.y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            exog=self.exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.result = self.model.fit(maxiter=1000, disp=False)

        self.pred_train = self.result.predict(
            start=self.y_train.index[0],
            end=self.y_train.index[-1],
            exog=self.exog_train
        )
        if self.log:
            self.pred_train = np.exp(self.pred_train)
            self.y_train = np.exp(self.y_train)

        return self

    def predict(self, y_test, exog_test=None):
        self.y_test = np.log(y_test) if self.log else y_test
        self.exog_test = exog_test

        self.pred_test = self.result.predict(
            start=self.y_test.index[0],
            end=self.y_test.index[-1],
            exog=self.exog_test
        )
        if self.log:
            self.pred_test = np.exp(self.pred_test)
            self.y_test = np.exp(self.y_test)

        return self.pred_test

    def score(self, metric='rmse'):
        if not hasattr(self, 'pred_test'):
            raise RuntimeError("First run `.predict()`")

        y_true = self.y_test
        y_pred = self.pred_test

        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            raise ValueError("Use: 'rmse', 'mae', 'mape'.")

    def report(self):
        if not hasattr(self, 'pred_test'):
            raise RuntimeError("First run `.predict()`")

        print(f"MAE test: {mean_absolute_error(self.y_test, self.pred_test):.2f}")
        print(f"RMSE test: {np.sqrt(mean_squared_error(self.y_test, self.pred_test)):.2f}")
        print(f"MAPE test: {np.mean(np.abs((self.y_test - self.pred_test) / self.y_test)) * 100:.2f}%")
        print("---")
        print(f"MAE train: {mean_absolute_error(self.y_train, self.pred_train):.2f}")
        print(f"RMSE train: {np.sqrt(mean_squared_error(self.y_train, self.pred_train)):.2f}")
        print(f"MAPE train: {np.mean(np.abs((self.y_train - self.pred_train) / self.y_train)) * 100:.2f}%")

    def plot_predictions(self, title='ARIMA Predictions'):
        if not hasattr(self, 'pred_test'):
            raise RuntimeError("First run `.predict()`")

        plt.figure(figsize=(10, 6))
        plt.plot(self.y_train.index, self.y_train, label='Train')
        plt.plot(self.y_test.index, self.y_test, label='Test Real', color='red')
        plt.plot(self.y_test.index, self.pred_test, label='Test Predicted', color='blue')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()
