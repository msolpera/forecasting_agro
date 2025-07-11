import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def check_series_validity(series, freq='MS'):
    """
    Verifica que una serie temporal:
    1. No tenga valores nulos.
    2. Sea continua en el tiempo (sin fechas faltantes).

    Parámetros:
    - series: pd.Series con índice datetime o period
    - freq: frecuencia esperada ('D', 'M', 'Q', etc.)

    Retorna:
    - True si es válida, False si no.
    - Lista de problemas encontrados.
    """

    issues = []

    # Convertir índice a datetime si es necesario
    if not series.index.inferred_type.startswith('datetime'):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            issues.append(f"Error al convertir índice a datetime: {e}")
            return False, issues

    # Chequear valores nulos
    if series.isnull().any().iloc[0]:
        issues.append("La serie contiene valores nulos.")

    # Chequear continuidad temporal
    full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq=freq)
    if not series.index.equals(full_index):
        missing_dates = full_index.difference(series.index)
        issues.append(f"La serie no es continua. Fechas faltantes: {len(missing_dates)}")

    is_valid = len(issues) == 0
    return is_valid, issues


def plot_time_series_resamples(series, title=''):
    """
    Genera tres subplots de una serie temporal:
    - Original (mensual)
    - Trimestral (quarter)
    - Anual (year)

    Parámetros:
    - series: pd.Series con índice datetime o period
    - title: título general de la figura
    """
    if not series.index.inferred_type.startswith('datetime'):
        series = series.copy()
        series.index = pd.to_datetime(series.index)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(title or 'Series Temporal: Mensual, Trimestral y Anual', fontsize=16)

    # Serie original (mensual)
    axs[0].plot(series, label='Mensual', color='steelblue')
    axs[0].set_title('Mensual')
    axs[0].grid(True)

    # Resample trimestral
    quarterly = series.resample('QE').mean()
    axs[1].plot(quarterly, label='Trimestral', color='darkorange')
    axs[1].set_title('Trimestral')
    axs[1].grid(True)

    # Resample anual
    annual = series.resample('YE').mean()
    axs[2].plot(annual, label='Anual', color='seagreen')
    axs[2].set_title('Anual')
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()