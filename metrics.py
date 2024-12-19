from typing import List, Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str] = ["mae", "mse", "rmse", "mape", "r2"],
) -> Dict[str, float]:
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
    available_metrics = {"mae", "mse", "rmse", "mape", "r2"}
    unsupported_metrics = set(metrics) - available_metrics
    if unsupported_metrics:
        raise ValueError(f"Неподдерживаемые метрики: {unsupported_metrics}")

    results = {}

    # MAE
    if "mae" in metrics:
        results["mae"] = mean_absolute_error(y_true, y_pred)

    # MSE
    if "mse" in metrics:
        results["mse"] = mean_squared_error(y_true, y_pred)

    # RMSE
    if "rmse" in metrics:
        results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE
    if "mape" in metrics:
        epsilon = 1e-10  # Чтобы избежать деления на ноль
        results["mape"] = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # R^2
    if "r2" in metrics:
        results["r2"] = r2_score(y_true, y_pred)

    return results


# @title residuals_analysis
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from typing import Any, Dict


def residuals_analysis(
    y_true: np.ndarray, y_pred: np.ndarray, dates: pd.DatetimeIndex
) -> Dict[str, Any]:
    """
    Проводит анализ остатков модели.

    Анализ включает:
    - Проверку на нормальность распределения (Shapiro-Wilk test)
    - Тест на автокорреляцию (Ljung-Box test)
    - Проверку гомоскедастичности (Breusch-Pagan test)
    - Базовую статистику остатков
    - Визуализацию QQ-plot и ACF

    Args:
        y_true (np.ndarray): Истинные значения
        y_pred (np.ndarray): Предсказанные значения
        dates (pd.DatetimeIndex): Индекс дат для временного анализа

    Returns:
        Dict[str, Any]: Словарь с результатами тестов и статистиками

    Example:
        >>> results = residuals_analysis(y_test, predictions, test.index)
        >>> print(f"Shapiro test p-value: {results['shapiro_pvalue']:.4f}")
    """
    residuals = y_true - y_pred

    # Проверяем, что размер дат соответствует остаткам
    if len(dates) != len(residuals):
        raise ValueError("Размер массива дат должен совпадать с размером остатков.")

    # Базовая статистика остатков
    residual_stats = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "min": np.min(residuals),
        "max": np.max(residuals),
    }

    # Тест на нормальность (Shapiro-Wilk test)
    shapiro_stat, shapiro_pvalue = shapiro(residuals)

    # Тест на автокорреляцию (Ljung-Box test)
    max_lag = min(len(residuals) // 2, 20)  # Лаги: не больше половины длины данных
    lb_test = acorr_ljungbox(residuals, lags=max_lag, return_df=True)
    lb_stat = lb_test["lb_stat"].iloc[-1]  # Значение статистики для максимального лага
    lb_pvalue = lb_test["lb_pvalue"].iloc[-1]  # Значение p-value для максимального лага

    # Тест на гомоскедастичность (Breusch-Pagan test)
    exog = sm.add_constant(np.arange(len(residuals)))
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, exog)

    # Визуализация
    plt.figure(figsize=(8, 6))

    # QQ-plot
    plt.subplot(2, 2, 1)
    qqplot(residuals, line="s", ax=plt.gca())
    plt.title("QQ-plot of Residuals")

    # ACF Plot
    plt.subplot(2, 2, 2)
    sm.graphics.tsa.plot_acf(residuals, lags=min(20, len(residuals) - 1), ax=plt.gca())
    plt.title("ACF of Residuals")

    # Временной ряд остатков
    plt.subplot(2, 1, 2)
    plt.plot(dates, residuals, label="Residuals", marker="o")
    plt.axhline(0, linestyle="--", color="red", alpha=0.7)
    plt.title("Residuals Over Time")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Результаты анализа
    results = {
        "residual_stats": residual_stats,
        "shapiro_stat": shapiro_stat,
        "shapiro_pvalue": shapiro_pvalue,
        "ljungbox_stat": lb_stat,
        "ljungbox_pvalue": lb_pvalue,
        "breuschpagan_stat": bp_stat,
        "breuschpagan_pvalue": bp_pvalue,
    }

    return results


# @title explain_residuals_analysis
def explain_residuals_analysis(results: Dict[str, Any], alpha: int = 0.05) -> None:
    """
    Объясняет результаты анализа остатков модели на основе переданных статистик.

    Args:
        results (Dict[str, Any]): Словарь с результатами анализа остатков
        alpha (int): Уровень значимости для t-тестов

    Example:
        >>> explain_residuals_analysis(results)
    """

    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # Получение статистик из результатов с преобразованием типов
    residual_stats = results.get("residual_stats", {})
    shapiro_pvalue = to_float(results.get("shapiro_pvalue"))
    ljungbox_pvalue = to_float(results.get("ljungbox_pvalue"))
    breuschpagan_pvalue = to_float(results.get("breuschpagan_pvalue"))

    print()

    # Базовая статистика остатков
    print("1. Базовая статистика остатков:")
    print(f"- Среднее значение: {residual_stats.get('mean', 'Нет данных'):.4f}")
    print(f"- Стандартное отклонение: {residual_stats.get('std', 'Нет данных'):.4f}")
    print(f"- Минимум: {residual_stats.get('min', 'Нет данных'):.4f}")
    print(f"- Максимум: {residual_stats.get('max', 'Нет данных'):.4f}")

    # Проверка на нормальность распределения
    print("\n2. Проверка на нормальность распределения (Shapiro-Wilk тест):")
    if shapiro_pvalue is not None:
        if shapiro_pvalue < alpha:
            print(
                f"- Остатки НЕ НОРМАЛЬНО распределены (p-value: {shapiro_pvalue:.4f})."
            )
            print(
                "  Это может указывать на то, что модель не идеально описывает данные."
            )
        else:
            print(f"- Остатки НОРМАЛЬНО распределены (p-value: {shapiro_pvalue:.4f}).")
            print("  Это хороший признак, остатки распределены случайно.")
    else:
        print("- Данные для теста отсутствуют.")

    # Проверка на автокорреляцию (Ljung-Box тест)
    print("\n3. Проверка на автокорреляцию (Ljung-Box тест):")
    if ljungbox_pvalue is not None:
        if ljungbox_pvalue < alpha:
            print(f"- Остатки имеют автокорреляцию (p-value: {ljungbox_pvalue:.4f}).")
            print("  Модель могла не учесть важные временные зависимости.")
        else:
            print(
                f"- Остатки НЕ имеют автокорреляции (p-value: {ljungbox_pvalue:.4f})."
            )
            print("  Хороший признак: модель учла временные зависимости.")
    else:
        print("- Данные для теста отсутствуют.")

    # Проверка гомоскедастичности (Breusch-Pagan тест)
    print("\n4. Проверка на гомоскедастичность (Breusch-Pagan тест):")
    if breuschpagan_pvalue is not None:
        if breuschpagan_pvalue < alpha:
            print(f"- Остатки НЕ гомоскедастичны (p-value: {breuschpagan_pvalue:.4f}).")
            print(
                "  Это может означать, что дисперсия остатков изменяется со временем."
            )
        else:
            print(f"- Остатки гомоскедастичны (p-value: {breuschpagan_pvalue:.4f}).")
            print("  Это хороший признак: дисперсия остатков постоянна.")
    else:
        print("- Данные для теста отсутствуют.")
