import joblib
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def create_features(df, target_col, lags=[1], add_month_dummies=True):
    df_feat = pd.DataFrame(index=df.index)
    df_feat[target_col] = df[target_col]
    
    # Lags
    for lag in lags:
        df_feat[f'lag_{lag}'] = df[target_col].shift(lag)
    
    df_feat['lag_12']=df[target_col].shift(12)
    df_feat['diff_seas'] = df[target_col] - df[target_col].shift(12)
    df_feat['diff_price'] = df[target_col] - df[target_col].shift(1)
    
    # Estacionalidad 
    if add_month_dummies:
        df_feat['month'] = df.index.month
        df_feat = pd.get_dummies(df_feat, columns=['month'], drop_first=True)
        df_feat.astype(float)
    return df_feat.dropna()


def forecasting_recursive(model, df, target_col, steps_ahead=5, output_csv=None):
    df_extended = df[[target_col]].copy()
    
    for step in range(steps_ahead):
        features = create_features(df_extended, target_col, lags=[1], add_month_dummies=True)
        last_features = features.iloc[-1:].drop(columns=[target_col])
        y_pred = model.predict(last_features, target_col, test=False)[0]
        next_date = df_extended.index[-1] + pd.DateOffset(months=1)  
        df_extended.loc[next_date] = np.round(y_pred,2)
    
    if output_csv:
        df_extended.to_csv(output_csv)
        print(f'Forecast saved to {output_csv}')
    
    return df_extended

def plot_forecasts_with_history(df_forecasted_dict, target_col, forecast_horizon):

    plt.plot(df_forecasted_dict[:-forecast_horizon].index, df_forecasted_dict[target_col][:-forecast_horizon], label='Histórico', color='blue')
    plt.plot(df_forecasted_dict[-forecast_horizon:].index, df_forecasted_dict[target_col][-forecast_horizon:], label='Forecast', linestyle='--', color='orange')
    cross_date = df_forecasted_dict.index[-forecast_horizon]
    plt.axvline(cross_date, color='gray', linestyle=':', label='Inicio Forecast')

    plt.title(f'{target_col.upper()} - Historic vs Forecast')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()