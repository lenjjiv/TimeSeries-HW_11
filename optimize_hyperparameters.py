from typing import Dict, Any, Callable
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    get_model: Callable,
    model_kwargs: Dict = None,
    n_trials: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 3,  # Добавляем кросс-валидацию
) -> Dict[str, Any]:
    """
    Универсальная функция оптимизации гиперпараметров моделей машинного обучения.

    Функция использует байесовскую оптимизацию (Optuna) для поиска оптимальных
    гиперпараметров любой модели машинного обучения. Включает валидацию через
    кросс-валидацию для более надежной оценки качества.

    Параметры:
        X (pd.DataFrame): Матрица признаков
        y (pd.Series): Целевая переменная
        get_model (Callable): Функция создания модели, принимающая trial и model_kwargs
        model_kwargs (Dict, optional): Дополнительные параметры для модели
        n_trials (int): Количество итераций оптимизации
        test_size (float): Размер валидационной выборки
        random_state (int): Значение для воспроизводимости результатов
        cv_folds (int): Количество фолдов для кросс-валидации

    Возвращает:
        Dict[str, Any]: Словарь с лучшими параметрами и историей оптимизации
    """
    model_kwargs = model_kwargs or {}

    def objective(trial):
        # Создаем модель с текущими параметрами
        model = get_model(trial, model_kwargs)

        # Проводим кросс-валидацию
        cv_scores = cross_val_score(
            model, X, y, cv=cv_folds, scoring="neg_mean_squared_error", n_jobs=-1
        )

        # Возвращаем среднее значение MSE по фолдам
        return -np.mean(cv_scores)

    # Создаем и запускаем оптимизацию
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Проводим финальную валидацию на отложенной выборке
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    final_model = get_model(study.best_trial, model_kwargs)
    final_model.fit(X_train, y_train)
    test_score = mean_squared_error(y_test, final_model.predict(X_test))

    return {
        "best_params": study.best_params,
        "study": study,
        "test_score": test_score,
        "best_model": final_model,
    }
