import mlflow
import pandas as pd
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix
from imblearn.pipeline import Pipeline
from train_val_test_split_A import X_train, X_val, X_test, y_train, y_val, y_test




def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 650),
        'max_depth': trial.suggest_int('max_depth', 8, 11),
        'min_samples_split': trial.suggest_int('min_samples_split', 225, 525),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 25, 100),
        'criterion': 'entropy',
        'max_features': 'sqrt',
        'bootstrap': False,
        'random_state': 42,
        'n_jobs': -1
    }

    # Создание модели
    model = RandomForestClassifier(**params)

    # Применяем SMOTE для балансировки
    smote = SMOTE(
        sampling_strategy='not majority',
        k_neighbors=3,
        random_state=42
    )
    pipeline = Pipeline(steps=[('smote', smote), ('classifier', model)])

    pipeline.fit(X_train, y_train)

    # Прогнозируем на валидационном наборе данных
    y_val_pred = pipeline.predict(X_val)

    # Рассчитываем recall
    recall = recall_score(y_val, y_val_pred)

    return recall


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=12, show_progress_bar=True)
best_params = study.best_params


# Логирование в MLflow
mlflow.set_experiment(f'Model_A')

with mlflow.start_run(run_name='RFC_with_smote'):
    # Установка тегов
    mlflow.set_tags({
        'model_type': 'RandomForest',
        'optimization_metric': 'recall',
        'balancing_method': 'smote',
        'sampling_strategy': 'default'
    })

    mlflow.log_params({
        **best_params,
        'criterion': 'entropy',
        'max_features': 'sqrt',
        'bootstrap': False,
        'optimized_metric': 'recall'
    })

    # Обучение финальной модели с лучшими параметрами
    model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    smote = SMOTE(random_state=42)
    pipeline = Pipeline(steps=[('smote', smote), ('classifier', model)])

    pipeline.fit(X_train, y_train)

    # Регистрация модели в MLflow
    mlflow.sklearn.log_model(pipeline, name='model')

    # Прогнозирование на тестовых данных
    y_test_pred = pipeline.predict(X_test)

    # Рассчитываем recall
    recall = recall_score(y_test, y_test_pred)

    # Логирование метрики recall в MLflow
    mlflow.log_metrics({'recall': recall})

    classes = sorted(y_test.unique())
    recall_per_class = recall_score(y_test, y_test_pred, average=None, labels=classes)

    class_counts = y_test.value_counts().sort_index()
    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    metrics_df = pd.DataFrame({
        'class': classes,
        'count': class_counts.values,
        'accuracy': class_accuracy,
        'recall': recall_per_class
    })

    metrics_df.to_csv('class_metrics.csv', index=False)

    mlflow.log_artifact('class_metrics.csv')
