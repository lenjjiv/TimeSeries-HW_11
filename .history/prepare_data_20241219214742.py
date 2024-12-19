
# @title create_calendar_features
from typing import List

def create_calendar_features(
        df: pd.DataFrame,
        holidays: List[str] = [
            "2024-01-01",
            "2024-07-04",
            "2024-12-25",
        ]
) -> pd.DataFrame:
    """
    Создает календарные признаки на основе временного индекса DataFrame.

    Создает следующие признаки:
    - час дня (0-23)
    - день недели (0-6)
    - месяц (1-12)
    - день месяца (1-31)
    - квартал (1-4)
    - признак выходного дня
    - признак праздничного дня (на основе пользовательского списка праздников)
    - сезонные циклические признаки (sin и cos для часа, дня недели и месяца)

    Args:
        df (pd.DataFrame): DataFrame с datetime индексом
        holidays (list): Список дат праздников в формате 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: DataFrame с добавленными календарными признаками

    Example:
        >>> df = create_calendar_features(df)
        >>> df.columns
        ['AEP_MW', 'hour', 'day_of_week', 'month', 'day_of_month',...]
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame должен иметь datetime индекс.")

    # Основные календарные признаки
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    df['quarter'] = df.index.quarter

    # Признак выходного дня (суббота и воскресенье)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Обработка праздников
    holidays = pd.to_datetime(holidays)
    df['is_holiday'] = df.index.isin(holidays).astype(int)

    # Циклические признаки (час, день недели, месяц)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

# @title load_and_prepare_data
def load_and_prepare_data(filepath: str, date_column: str) -> pd.DataFrame:
    """
    Загружает и подготавливает временной ряд из CSV файла для дальнейшего анализа.

    Функция выполняет следующие операции:
    - Загружает данные из CSV
    - Преобразует столбец с датами в datetime
    - Устанавливает дату как индекс
    - Проверяет и обрабатывает пропущенные значения
    - Сортирует данные по времени
    - Проверяет равномерность временного ряда

    Args:
        filepath (str): Путь к CSV файлу с данными
        date_column (str): Название столбца с датами

    Returns:
        pd.DataFrame: Подготовленный DataFrame с datetime индексом

    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если в данных обнаружены критические проблемы

    Example:
        >>> df = load_and_prepare_data('hw_AEP.csv', 'Datetime')
        >>> df.head()
        Datetime               AEP_MW
        2004-01-01 01:00:00  13478.0
        2004-01-01 02:00:00  13213.0
        ...
    """
    import pandas as pd

    # 1. Load the data
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    # 2. Convert date_column to datetime
    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' not found in the dataset.")

    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

    if data[date_column].isnull().any():
        raise ValueError("Some date values could not be converted to datetime.")

    # 3. Set the datetime column as index
    data.set_index(date_column, inplace=True)

    # 4. Handle missing values
    if data.isnull().any().any():
        data = data.fillna(method='ffill').fillna(method='bfill')
        if data.isnull().any().any():
            raise ValueError("Missing values remain after forward/backward filling.")

    # 5. Sort by datetime
    data.sort_index(inplace=True)

    # 6. Check if the time series is evenly spaced
    time_diffs = data.index.to_series().diff().dropna()
    if not (time_diffs == time_diffs.iloc[0]).all():
        raise ValueError("The time series is not evenly spaced.")

    return data

# @title train_test_split_ts
from typing import Tuple
import pandas as pd

def train_test_split_ts(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделяет временной ряд на обучающую и тестовую выборки с сохранением временной структуры.
    - Сохраняет временную последовательность
    - Не перемешивает данные
    - Берет последние test_size% наблюдений в тестовую выборку

    Args:
        df (pd.DataFrame): Исходный DataFrame с временным рядом
        test_size (float): Доля тестовой выборки (0 < test_size < 1)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (обучающая выборка, тестовая выборка)

    Raises:
        ValueError: Если test_size не в интервале (0, 1)

    Example:
        >>> train, test = train_test_split_ts(df, test_size=0.2)
        >>> print(f"Train size: {len(train)}, Test size: {len(test)}")
    """
    # Проверяем значение test_size
    if not (0 < test_size < 1):
        raise ValueError("test_size должен быть в интервале (0, 1).")

    # Вычисляем размер тестовой выборки
    n_test = int(len(df) * test_size)

    # Разделяем данные
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]

    return train, test


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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex
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
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals)
    }

    # Тест на нормальность (Shapiro-Wilk test)
    shapiro_stat, shapiro_pvalue = shapiro(residuals)

    # Тест на автокорреляцию (Ljung-Box test)
    max_lag = min(len(residuals) // 2, 20)  # Лаги: не больше половины длины данных
    lb_test = acorr_ljungbox(residuals, lags=max_lag, return_df=True)
    lb_stat = lb_test['lb_stat'].iloc[-1]  # Значение статистики для максимального лага
    lb_pvalue = lb_test['lb_pvalue'].iloc[-1]  # Значение p-value для максимального лага

    # Тест на гомоскедастичность (Breusch-Pagan test)
    exog = sm.add_constant(np.arange(len(residuals)))
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, exog)

    # Визуализация
    plt.figure(figsize=(8, 6))

    # QQ-plot
    plt.subplot(2, 2, 1)
    qqplot(residuals, line='s', ax=plt.gca())
    plt.title('QQ-plot of Residuals')

    # ACF Plot
    plt.subplot(2, 2, 2)
    sm.graphics.tsa.plot_acf(residuals, lags=min(20, len(residuals) - 1), ax=plt.gca())
    plt.title('ACF of Residuals')

    # Временной ряд остатков
    plt.subplot(2, 1, 2)
    plt.plot(dates, residuals, label='Residuals', marker='o')
    plt.axhline(0, linestyle='--', color='red', alpha=0.7)
    plt.title('Residuals Over Time')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Результаты анализа
    results = {
        'residual_stats': residual_stats,
        'shapiro_stat': shapiro_stat,
        'shapiro_pvalue': shapiro_pvalue,
        'ljungbox_stat': lb_stat,
        'ljungbox_pvalue': lb_pvalue,
        'breuschpagan_stat': bp_stat,
        'breuschpagan_pvalue': bp_pvalue
    }

    return results