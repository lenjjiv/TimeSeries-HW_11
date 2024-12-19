


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