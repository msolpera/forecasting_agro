import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

class XGBoostPipeline:
    def __init__(self, metric='rmse'):
        self.metric = metric
        self.model = None
        self.params = {
                    'max_depth': [3, 5],
                    'min_child_weight': [1, 5],
                    'gamma': [0, 0.5, 1],
                    'subsample': [0.6, 0.8],
                    'colsample_bytree': [0.8],
                    'learning_rate': [0.1],
                    'reg_alpha':[0.5], 'reg_lambda':[1.0]
                    }


    def fit(self, train_feat, target_col):
        X_train = train_feat.drop(columns=target_col)
        y_train = train_feat[target_col]

        # Cross-validation + GridSearch
        tscv = TimeSeriesSplit(n_splits=5)
        
        xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
        grid = GridSearchCV(xgb, param_grid=self.params, cv=tscv, scoring='neg_mean_squared_error', verbose=1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        self.model = best_model

        # Guardar predicciones en train
        self.y_pred_train = self.model.predict(X_train)
        self.y_train_true = train_feat[target_col]

        return self

    def predict(self, test_feat, target_col, test=True):
        if test:
            self.y_test_true = test_feat[target_col]
            X_test = test_feat.drop(columns=target_col)
        else:
            X_test = test_feat
        
        y_pred_test = self.model.predict(X_test)
        self.y_pred_test=y_pred_test

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

    def plot_predictions(self, title="XGBoost Forecast vs Real"):
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