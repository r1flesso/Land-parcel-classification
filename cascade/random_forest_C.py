import mlflow
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from train_val_test_split_C import X_train, X_val, X_test, y_train, y_val, y_test, dict_




def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 90, 610),
        'max_depth': trial.suggest_int('max_depth', 9, 11),
        'min_samples_split': trial.suggest_int('min_samples_split', 225, 545),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 90),
        'criterion': 'entropy',
        'max_features': 'sqrt',
        'class_weight': dict_,
        'bootstrap': False,
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestClassifier(**params)

    model.fit(X_train, y_train)

    y_val_predict = model.predict(X_val)

    score = accuracy_score(y_val, y_val_predict)

    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)
best_params = study.best_params


mlflow.set_experiment(f'Model_C')

with mlflow.start_run(run_name='RFC_with_custom_dict'):
    mlflow.set_tags({
        'model_type': 'RandomForest',
        'optimization_metric': 'accuracy',
        'task_type': 'territory_classification'
    })

    mlflow.log_params({
        **best_params,
        'criterion': 'entropy',
        'max_features': 'sqrt',
        'bootstrap': False,
        'optimized_metric': 'accuracy',
        'class_weight': dict_
    })

    model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, name='model')

    y_test_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_test_pred)

    mlflow.log_metrics({'accuracy': accuracy})

    classes = sorted(y_test.unique())

    class_counts = y_test.value_counts().sort_index()
    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    metrics_df = pd.DataFrame({
        'class': classes,
        'count': class_counts.values,
        'accuracy': class_accuracy
    })

    metrics_df.to_csv('class_metrics.csv', index=False)

    mlflow.log_artifact('class_metrics.csv')
