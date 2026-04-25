import mlflow.sklearn
import pandas as pd
from train_val_test_split import X_train, X_val, X_test, y_train, y_val, y_test
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score,\
    cohen_kappa_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import itertools




# Класс для обучения и оптимизации RandomForest
class RandomForestClassifier_Model():
    def __init__(
        self, X_train, X_val, X_test, y_train, y_val, y_test, max_features, bootstrap, metric
    ):
        # Инициализация данных и параметров
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.max_features = max_features  # Количество признаков для разделения
        self.bootstrap = bootstrap  # Использовать ли бутстрап выборки
        self.metric = metric  # Метрика для оптимизации

    def objective_optuna(self):
        # Функция вычисления выбранной метрики
        def calculate_metric_score(y_true, y_pred, y_pred_proba=None, metric=None):
            try:
                if metric == 'f1':
                    return f1_score(y_true, y_pred, average='weighted', zero_division=0)
                elif metric == 'recall':
                    return recall_score(y_true, y_pred, average='weighted', zero_division=0)
                elif metric == 'precision':
                    return precision_score(y_true, y_pred, average='weighted', zero_division=0)
                elif metric == 'cohen_kappa':
                    return cohen_kappa_score(y_true, y_pred)
                elif metric == 'matthews_corrcoef':
                    return matthews_corrcoef(y_true, y_pred)
                elif metric == 'roc_auc_score':
                    # Использую roc_auc_score с multi_class='ovo' (One-vs-One),
                    # который вычисляет AUC попарно для всех комбинаций классов.
                    # Этот подход часто лучше подходит для несбалансированных данных.
                    if y_pred_proba is not None:
                        return roc_auc_score(
                            y_true, y_pred_proba, average='weighted', multi_class='ovo'
                        )
                    else:
                        return 0.0
            except:
                return 0.0  # Возвращаем 0 при ошибке

        # Целевая функция для Optuna
        def objective(trial):
            # Гиперпараметры для оптимизации
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 275),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 50, 1500),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 950),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'max_samples': trial.suggest_float('max_samples', 0.7, 1.0) if self.bootstrap else None,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }

            model = RandomForestClassifier(**params)  # Создание модели

            model.fit(self.X_train, self.y_train)  # Обучение

            # В зависимости от метрики используем разные предсказания
            if self.metric == 'roc_auc_score':
                # Для roc_auc нужны вероятности
                y_val_proba = model.predict_proba(self.X_val)
                score = calculate_metric_score(self.y_val, None, y_val_proba, self.metric)
            else:
                # Для остальных метрик достаточно меток классов
                y_val_pred = model.predict(self.X_val)
                score = calculate_metric_score(self.y_val, y_val_pred, None, self.metric)

            if score is None:
                return 0.0

            return score


        study = optuna.create_study(direction='maximize')  # Создание исследования Optuna
        study.optimize(objective, n_trials=24, show_progress_bar=True)  # Запуск оптимизации

        best_params = study.best_params  # Лучшие параметры


        # Создаем эксперимент MLflow, в который будем логировать теги, метрики, модель т.д.
        mlflow.set_experiment('Random_Forest_Classifier')

        # Создаем запуск MLflow, указываем имя run
        with mlflow.start_run(
                run_name=f'RFC_with_{self.max_features}_n_{self.bootstrap}_bootstrap_n_{self.metric}'
        ):
            # Логируем параметры модели
            mlflow.log_params({
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'optimized_metric': self.metric,
                **best_params
            })

            # Обучение финальной модели с лучшими параметрами
            forest = RandomForestClassifier(
                **best_params,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

            forest.fit(self.X_train, self.y_train)

            # Предсказание на тестовой выборке
            predict_labels = forest.predict(self.X_test)
            predict_proba = forest.predict_proba(self.X_test)  # Вероятности для roc_auc

            # Вычисление всех метрик
            f1 = f1_score(y_test, predict_labels, average='weighted')
            rec = recall_score(y_test, predict_labels, average='weighted')
            pre = precision_score(y_test, predict_labels, average='weighted')
            kappa = cohen_kappa_score(y_test, predict_labels)
            mcc = matthews_corrcoef(y_test, predict_labels)
            roc_auc = roc_auc_score(y_test, predict_proba, average='weighted', multi_class='ovo')

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
            print(f'===== {self.max_features}_n_{self.bootstrap}_n_{self.metric} =====')
            print(f'F1-score: {f1:.4f}')
            print(f'Recall: {rec:.4f}')
            print(f'Precision: {pre:.4f}')
            print(f'Cohen Kappa: {kappa:.4f}')
            print(f'Matthews Correlation: {mcc:.4f}')
            print(f'Roc-Auc-score: {roc_auc:.4f}')

            # Регистрация модели
            mlflow.sklearn.log_model(forest, name='model')

            # Установка тегов
            mlflow.set_tags({
                'model_type': 'RandomForest',
                'optimization_metric': self.metric,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'task_type': 'territory_classification',
            })

            classes = sorted(y_test.unique())
            precision_per_class = precision_score(y_test, predict_labels, average=None, labels=classes)
            recall_per_class = recall_score(y_test, predict_labels, average=None, labels=classes)
            f1_per_class = f1_score(y_test, predict_labels, average=None, labels=classes)


            # ROC AUC по классам (One-vs-Rest для каждого класса)
            roc_auc_per_class = []
            for i, class_label in enumerate(classes):
                try:
                    y_true_binary = (y_test == class_label).astype(int)
                    if len(np.unique(y_true_binary)) > 1:
                        auc = roc_auc_score(y_true_binary, predict_proba[:, i])
                    else:
                        auc = 0.0
                except:
                    auc = 0.0
                roc_auc_per_class.append(auc)

            class_counts = y_test.value_counts().sort_index()

            cm = confusion_matrix(y_test, predict_labels, labels=classes)
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


# Определяем комбинации bootstrap и metric для перебора
bootstrap_list = [True, False]
metrics = ['f1', 'recall', 'precision', 'cohen_kappa', 'matthews_corrcoef', 'roc_auc_score']

# Основной блок выполнения
if __name__ == '__main__':
    # Бежим по циклу, перебираем каждые комбинации bootstrap и metric
    for bootstrap, metric in itertools.product(bootstrap_list, metrics):
        # Создаем объект класса с текущими параметрами
        Obj = RandomForestClassifier_Model(
            X_train, X_val, X_test, y_train, y_val, y_test, 'sqrt', bootstrap, metric
        )
        # Запускаем оптимизацию и логирование
        Obj.objective_optuna()
