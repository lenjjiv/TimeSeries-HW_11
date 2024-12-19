
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