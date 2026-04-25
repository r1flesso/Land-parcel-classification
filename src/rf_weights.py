import mlflow.sklearn
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score,\
    cohen_kappa_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from train_val_test_split import X_train, X_val, X_test, y_train, y_val, y_test
from Custom_dicts import log_inverse_frequency_dict, sqrt_weights




def objective(trial, custom_dict):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 190, 240),
        'max_depth': trial.suggest_int('max_depth', 9, 11),
        'min_samples_split': trial.suggest_int('min_samples_split', 417, 497),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 30, 85),
        'criterion': 'entropy',
        'max_features': 'sqrt',
        'bootstrap': False,
        'class_weight': custom_dict,
        'random_state': 42,
        'n_jobs': -1
    }

    # Создание модели
    model = RandomForestClassifier(**params)

    # Обучаем модель
    model.fit(X_train, y_train)

    # Прогнозируем на валидационном наборе данных
    y_val_proba = model.predict_proba(X_val)

    # Рассчитываем метрику
    score = roc_auc_score(
        y_val,
        y_val_proba,
        average='weighted',
        multi_class='ovo'
    )

    return score


dicts = ['log_inverse_frequency_dict', 'sqrt_weights']

for dict_ in dicts:
    current_dict = log_inverse_frequency_dict if dict_ == 'log_inverse_frequency_dict' \
        else sqrt_weights


    # Создаем обертку для передачи словаря в objective
    def wrapped_objective(trial):
        return objective(trial, current_dict)


    study = optuna.create_study(direction='maximize')
    # Передаем обернутую функцию
    study.optimize(wrapped_objective, n_trials=40, show_progress_bar=True)
    best_params = study.best_params


    # Логирование в MLflow
    mlflow.set_experiment(f'RFC_with_custom_dicts')

    with mlflow.start_run(run_name=f'RFC_with_{dict_}'):
        # Установка тегов
        mlflow.set_tags({
            'model_type': 'RandomForest',
            'optimization_metric': 'roc_auc_score',
            'task_type': 'territory_classification',
            'custom_dict': dict_
        })

        mlflow.log_params({
            **best_params,
            'criterion': 'entropy',
            'max_features': 'sqrt',
            'bootstrap': False,
            'class_weight_dict': str(current_dict),
            'optimized_metric': 'roc_auc_score'
        })

        # Обучение финальной модели с лучшими параметрами
        model = RandomForestClassifier(
            **best_params,
            criterion='entropy',
            max_features='sqrt',
            bootstrap=False,
            class_weight=current_dict,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, name='model')

        # Прогнозирование на тестовых данных
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        # Рассчитываем метрики
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        rec = recall_score(y_test, y_test_pred, average='weighted')
        pre = precision_score(y_test, y_test_pred, average='weighted')
        kappa = cohen_kappa_score(y_test, y_test_pred)
        mcc = matthews_corrcoef(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba, average='weighted', multi_class='ovo')

        # Логирование метрик в MLflow
        mlflow.log_metrics({
            'f1_score': f1,
            'recall': rec,
            'precision': pre,
            'cohen_kappa': kappa,
            'matthews_corrcoef': mcc,
            'roc_auc_score': roc_auc
        })

        # Вывод метрик в консоль
        print(dict_)
        print(f'F1-score: {f1:.4f}')
        print(f'Recall: {rec:.4f}')
        print(f'Precision: {pre:.4f}')
        print(f'Cohen Kappa: {kappa:.4f}')
        print(f'Matthews Correlation: {mcc:.4f}')
        print(f'Roc-Auc-score: {roc_auc:.4f}')

        classes = sorted(y_test.unique())
        precision_per_class = precision_score(y_test, y_test_pred, average=None, labels=classes)
        recall_per_class = recall_score(y_test, y_test_pred, average=None, labels=classes)
        f1_per_class = f1_score(y_test, y_test_pred, average=None, labels=classes)

        # Логирование метрик по классам
        roc_auc_per_class = []
        for i, class_label in enumerate(classes):
            try:
                y_true_binary = (y_test == class_label).astype(int)
                if len(np.unique(y_true_binary)) > 1:
                    auc = roc_auc_score(y_true_binary, y_test_proba[:, i])
                else:
                    auc = 0.0
            except:
                auc = 0.0
            roc_auc_per_class.append(auc)

        class_counts = y_test.value_counts().sort_index()

        cm = confusion_matrix(y_test, y_test_pred, labels=classes)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

        metrics_df = pd.DataFrame({
            'class': classes,
            'count': class_counts.values,
            'accuracy': class_accuracy,
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'roc_auc': roc_auc_per_class
        })

        metrics_df.to_csv('class_metrics.csv', index=False)

        mlflow.log_artifact('class_metrics.csv')
