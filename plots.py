# @title plot_forecast_results
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from typing import Optional


def plot_acf_with_intervals(
    series: pd.Series,
    nlags: int = 50,
    alpha: float = 0.05,
    width: int = 800,
    height: int = 500,
    plot_original_series: bool = True,
) -> None:
    """
    Визуализирует ACF и PACF с доверительными интервалами.

    Отображает:
    - Оригинальный временной ряд (опционально)
    - График ACF
    - График PACF

    Args:
        series (pd.Series): Временной ряд
        nlags (int): Количество лагов для расчета ACF и PACF
        alpha (float): Уровень значимости для доверительных интервалов
        width (int): Ширина графиков
        height (int): Высота графиков
        plot_original_series (bool): Флаг отображения оригинального временного ряда

    Returns:
        None: Отображает интерактивные графики с использованием plotly.express

    Example:
        >>> plot_acf_with_intervals(series, nlags=40, alpha=0.05)
    """
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from statsmodels.tsa.stattools import acf, pacf

    def plot_series(data, title, y_label):
        """Функция для визуализации временного ряда."""
        return px.line(
            x=data.index,
            y=data,
            title=title,
            labels={"x": "Time", "y": y_label},
            width=width,
            height=height,
        )

    def plot_bars(data, title, y_label, error_column):
        """Функция для построения столбчатых графиков ACF/PACF."""
        fig = px.bar(
            data,
            x="Time",
            y=y_label,
            error_y=error_column,
            range_y=[-1.2, 1.2],
            title=title,
            width=width,
            height=height,
        )
        fig.update_traces(error_y=dict(color="black", thickness=1, width=2))
        fig.update_layout(
            xaxis=dict(showgrid=True, gridcolor="white"),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="black",
                dtick=1,
                showgrid=True,
                gridcolor="white",
            ),
        )
        return fig

    # Отображение исходного временного ряда
    if plot_original_series:
        plot_series(series, "Оригинальный временной ряд", "Value").show()

    # Расчет ACF и PACF
    acf_values, confint_acf = acf(series, alpha=alpha, nlags=nlags, fft=False)
    pacf_values, confint_pacf = pacf(series, alpha=alpha, nlags=nlags)

    # Вычисление ошибок
    acf_df = pd.DataFrame(
        {
            "Time": np.arange(1, len(acf_values)),
            "ACF": acf_values[1:],
            "Error_ACF": confint_acf[1:, 1] - acf_values[1:],
        }
    )

    pacf_df = pd.DataFrame(
        {
            "Time": np.arange(1, len(pacf_values)),
            "PACF": pacf_values[1:],
            "Error_PACF": confint_pacf[1:, 1] - pacf_values[1:],
        }
    )

    # Построение графиков
    plot_bars(acf_df, "Autocorrelation Function (ACF)", "ACF", "Error_ACF").show()
    plot_bars(
        pacf_df, "Partial Autocorrelation Function (PACF)", "PACF", "Error_PACF"
    ).show()


def plot_forecast_results(
    y_true: pd.Series,
    y_pred: pd.Series,
    intervals: Optional[pd.DataFrame] = None,
    title: str = "Forecast Results",
) -> None:
    """
    Создает комплексную визуализацию результатов прогнозирования.

    Отображает:
    - Фактические значения
    - Прогноз
    - Доверительные интервалы (если предоставлены)
    - Метрики качества на графике

    Args:
        y_true (pd.Series): Фактические значения
        y_pred (pd.Series): Прогноз
        intervals (Optional[pd.DataFrame]): Доверительные интервалы
        title (str): Заголовок графика

    Returns:
        None: Отображает график

    Example:
        >>> plot_forecast_results(y_test, predictions, confidence_intervals,
        >>>                      'LightGBM Forecast Results')
    """
    # Метрики качества
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(8, 4))
    plt.plot(y_true.index, y_true, label="Actual", color="blue", linewidth=2)
    plt.plot(
        y_pred.index,
        y_pred,
        label="Forecast",
        color="orange",
        linestyle="--",
        linewidth=2,
    )

    # Отображение доверительных интервалов
    if intervals is not None:
        plt.fill_between(
            intervals.index,
            intervals["lower"],
            intervals["upper"],
            color="gray",
            alpha=0.2,
            label="Confidence Interval",
        )

    # Добавление метрик на график
    metrics_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}"
    plt.gca().text(
        0.02,
        0.95,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    # Настройки графика
    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
