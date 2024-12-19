# @title calculate_metrics
from typing import List, Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      metrics: List[str] = ['mae', 'mse', 'rmse', 'mape', 'r2']) -> Dict[str, float]:
    """
    Вычисляет набор метрик качества прогноза.

    Поддерживает следующие метрики:
    - MAE (Mean Absolute Error)
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R^2 (Coefficient of Determination)

    Args:
        y_true (np.ndarray): Истинные значения
        y_pred (np.ndarray): Предсказанные значения
        metrics (List[str]): Список метрик для расчета

    Returns:
        Dict[str, float]: Словарь {название метрики: значение}

    Raises:
        ValueError: Если запрошена неподдерживаемая метрика

    Example:
        >>> metrics = calculate_metrics(y_test, predictions, ['mae', 'rmse'])
        >>> print(f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
    """
    # Поддерживаемые метрики
    available_metrics = {'mae', 'mse', 'rmse', 'mape', 'r2'}
    unsupported_metrics = set(metrics) - available_metrics
    if unsupported_metrics:
        raise ValueError(f"Неподдерживаемые метрики: {unsupported_metrics}")

    results = {}

    # MAE
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true, y_pred)

    # MSE
    if 'mse' in metrics:
        results['mse'] = mean_squared_error(y_true, y_pred)

    # RMSE
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE
    if 'mape' in metrics:
        epsilon = 1e-10  # Чтобы избежать деления на ноль
        results['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # R^2
    if 'r2' in metrics:
        results['r2'] = r2_score(y_true, y_pred)

    return results